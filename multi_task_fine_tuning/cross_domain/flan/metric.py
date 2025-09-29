from typing import Dict, List
from t5.evaluation import metrics
import tqdm
import json
import os


def read_list(file,k):
    dic={}
    lines=open(file).readlines()
    for line in lines:
        line = line.strip()
        data = json.loads(line)
        if data['category'] not in dic.keys():
            dic[data['category']] = []
        tmpd=data[k]
        if data[k].endswith('</s>'):
            tmpd = data[k].split('</s>')[0]
        #if data['category'] in ['paraphrase','question_classification']:
           # tmpd = tmpd.split(' ')[0].strip(',')
        dic[data['category']].append(tmpd)
    return dic

# Multi-rouge/multi-bleu. When there are multiple references, we want to get the
# rouge score that is highest. According to the authors, this is how it was done
# in the GEM paper.
# Source: https://github.com/google/BIG-bench/blob/main/bigbench/api/task_metrics.py
def rouge_fn(targets: List[List[str]], predictions: List[str]) -> Dict[str, float]:
  """Computes ROUGE by taking the max ROUGE-N per reference and N."""
  # Following strategy from https://www.aclweb.org/anthology/W04-1013/.
  # Identify best reference per response and ROUGE type.
  rouge_types = ["rouge1", "rouge2", "rougeLsum"]
  max_references = {rouge_type: [] for rouge_type in rouge_types}
  for targ_for_resp, resp in tqdm.tqdm(zip(targets, predictions), total=len(targets)):
    # Compute individual scores per example/ref pair.
    resp_scores = [metrics.rouge([t], [resp]) for t in targ_for_resp]
    # Find best scoring references for generated output and ROUGE type.
    for rouge_type in rouge_types:
      best_score_index = max(range(len(resp_scores)), key=lambda x: resp_scores[x][rouge_type])
      best_ref = targ_for_resp[best_score_index]
      # Add the reference to the new reference list.
      max_references[rouge_type].append(best_ref)
  # Compute metric for each of the reference lists for a ref type.
  results = {}
  for rouge_type in rouge_types:
    results[rouge_type] = metrics.rouge(max_references[rouge_type], predictions)[rouge_type]
  return results

def rouge(targets, predictions):
    results = metrics.rouge(targets, predictions)
    return results

def get_result(targets, predictions, save, category=None):
    results = {}
    if category is not None:
       assert category in targets
       assert category in predictions
       result = rouge(targets[category], predictions[category])
       results[category] = result
    else:
        for k in targets.keys():
            result = rouge(targets[k], predictions[k])
            results[k] = result

    with open(save, 'w') as f:
        f.write(json.dumps(results))


method = "alora-r8-a16-3e4"
targets = read_list('data/flan_test_200_selected_nstrict_1.jsonl', 'output')
predictions = read_list(f'output/results/{method}/inference.jsonl', 'answer')

# client_category_dic = {
#    0: "common_sense", 1: "entailment", 2: "open_domain_qa", 
#    3: "paraphrase", 4: "reading_comprehension", 5: "sentiment", 
#    6: "summarization", 7: "text_formatting"
# }

get_result(targets, predictions, f'output/results/{method}/evaluation.json')