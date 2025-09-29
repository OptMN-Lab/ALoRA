import os

import fire
import numpy as np
import torch
import transformers
import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer,AutoTokenizer
from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass


def writeFile(s, path):
    with open(path,'a+',encoding='utf-8') as f1:
        f1.write(s+'\n')

# random seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(42)

def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights_path: str = "",
    lora_config_path: str= "", # provide only the file path, excluding the file name 'adapter_config.json'
    lora_base_weights_path: str = "",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    server_name: str = "127.0.0.1",
    share_gradio: bool = False,
    output_file: str="",
    test_file: str="",
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    gpu_count = torch.cuda.device_count()

    if not lora_weights_path.endswith(".bin"):
        if device == "cuda":
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            model = PeftModel.from_pretrained(
                model,
                lora_weights_path,
                torch_dtype=torch.float16,
            )
        elif device == "mps":
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
            model = PeftModel.from_pretrained(
                model,
                lora_weights_path,
                device_map={"": device},
                torch_dtype=torch.float16,
            )
        else:
            model = LlamaForCausalLM.from_pretrained(
                base_model, device_map={"": device}, low_cpu_mem_usage=True
            )
            model = PeftModel.from_pretrained(
                model,
                lora_weights_path,
                device_map={"": device},
            )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
        config = LoraConfig.from_pretrained(lora_config_path)
        if gpu_count < 3:
            print(gpu_count)
            lora_weights = torch.load(lora_weights_path, weights_only=True, map_location=lambda storage, loc: storage.cuda(0))
        else:
            lora_weights = torch.load(lora_weights_path, weights_only=True)
        model = PeftModel(model, config)
        set_peft_model_state_dict(model, lora_weights, "default")
        model.set_adapter('default')
        del lora_weights
    
    if lora_base_weights_path:
        if gpu_count < 3:
            lora_base_weights = torch.load(lora_base_weights_path, weights_only=True, map_location=lambda storage, loc: storage.cuda(0))
        else:
            lora_base_weights = torch.load(lora_weights_path, weights_only=True)
        missing, unexpected = model.load_state_dict(lora_base_weights, strict=False)
        assert len(unexpected) == 0

    #exit()

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    tokenizer.padding_side = "left"
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()


    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=80,
        stream_output=True,
        input_ids=None,
        **kwargs,
    ):
        if input_ids is not None:
            input_ids = input_ids.to(device)
            #print(input_ids)
        else:
            prompt = prompter.generate_prompt(instruction, input)
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        if len(generation_output.sequences) ==1:
            s = generation_output.sequences[0]
            output = tokenizer.decode(s)
            ans = prompter.get_response(output)
        else:
            s = generation_output.sequences.cpu()
            output = tokenizer.batch_decode(s)
            ans = [prompter.get_response(t).split('</s>')[0] for t in output]
        return ans

    lines = open(test_file).readlines()
    count = 0
    for i, line in enumerate(lines):
        line = line.strip()
        ques = json.loads(line)
        res = evaluate(ques['instruction'])

        tmp = {}
        tmp['text'] = ques['instruction']
        tmp['answer'] = res
        tmp['category'] = ques['category']
        writeFile(json.dumps(tmp, ensure_ascii=False), output_file)
        count = count + 1
        if count % 100 == 0:
            print('num:', count)
            print("Instruction:", tmp['text'])
            print("Response:", tmp['answer'])
            print("*****************************************************")
        # break



if __name__ == "__main__":
    fire.Fire(main)
