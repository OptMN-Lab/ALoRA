from peft import (
    set_peft_model_state_dict,
)
import torch
import os
from torch.nn.functional import normalize


def FedAvg(selected_clients_set, output_dir, local_dataset_len_dict, epoch):
    weights_array = normalize(
        torch.tensor([local_dataset_len_dict[client_id] for client_id in selected_clients_set],
                     dtype=torch.float32),
        p=1, dim=0)

    for k, client_id in enumerate(selected_clients_set):
        single_output_dir = os.path.join(output_dir, str(epoch), "local_output_{}".format(client_id),
                                         "pytorch_model.bin")
        single_weights = torch.load(single_output_dir, weights_only=True)
        single_weights = {k: v for k, v in single_weights.items() if 'lora_B' in k}
        single_weights = recover_delta(single_weights, target_module='lora_B')
        if k == 0:
            weighted_single_weights = {key: single_weights[key] * (weights_array[k]) for key in
                                       single_weights.keys()}
        else:
            weighted_single_weights = {key: weighted_single_weights[key] + single_weights[key] * (weights_array[k])
                                       for key in
                                       single_weights.keys()}

    weighted_single_weights = update_base(weighted_single_weights, output_dir, epoch)
    torch.save(weighted_single_weights, os.path.join(output_dir, str(epoch), "adapter_model.bin"))

    return


def recover_delta(weight_dict, target_module="lora_B"):
    nset = []
    for n, p in weight_dict.items():
        new_n = n.split('lora_')[0]
        if new_n not in nset:
            nset.append(new_n)
    new_weight_dict = {}
    for n in nset:
        mat1 = weight_dict[n + f"{target_module}1.weight"]
        mat2 = weight_dict[n + f"{target_module}2.weight"]
        mat = mat2 @ mat1
        new_weight_dict[n + f"{target_module}_base_default"] = mat
    return new_weight_dict


def update_base(weight_dict, output_dir, epoch):
    # previous base_B
    prev_weight_dict = None
    if epoch and os.path.exists(os.path.join(output_dir, str(epoch - 1), "adapter_model.bin")):
        prev_weight_dict = torch.load(os.path.join(output_dir, str(epoch - 1), "adapter_model.bin"), weights_only=True)
    if prev_weight_dict is not None:
        weight_dict = {key: weight_dict[key] + prev_weight_dict[key] for key in weight_dict.keys()}
    return weight_dict