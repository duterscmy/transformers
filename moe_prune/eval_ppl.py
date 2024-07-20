# -*- coding: utf-8 -*-
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import torch
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoTokenizer, T5Tokenizer, AutoConfig, AutoModelForCausalLM, LogitsProcessorList, LogitsProcessor
from typing import List, Optional
import random
import os
import json
# import shortuuid
import time

from transformers.models.qwen2_moe.expert_idx import *

import torch
import torch.nn.functional as F


def calculate_kl_divergence(probs_p, probs_q):
    """
    计算两个分布之间的KL散度
    :param probs_p: 真实分布的概率分布 (形状：[N, D])
    :param probs_q: 近似分布的概率分布 (形状：[N, D])
    :return: KL散度
    """
    epsilon = 1e-10
    probs_p = probs_p + epsilon
    probs_q = probs_q + epsilon
    kl_div = F.kl_div(probs_q.log(), probs_p, reduction='batchmean')  # 计算KL散度
    return kl_div

def calculate_js_divergence(logits_p, logits_q):
    """
    计算两个分布之间的Jensen-Shannon散度
    :param logits_p: 真实分布的logits (形状：[N, D])
    :param logits_q: 近似分布的logits (形状：[N, D])
    :return: JS散度
    """
    p = F.softmax(logits_p, dim=-1)  # 将logits转化为概率分布
    q = F.softmax(logits_q, dim=-1)  # 将logits转化为概率分布

    m = 0.5 * (p + q)
    kl_pm = calculate_kl_divergence(p, m)
    kl_qm = calculate_kl_divergence(q, m)
    js_div = 0.5 * (kl_pm + kl_qm)
    print("kl per sample {} {} {}".format(kl_pm, kl_qm, js_div))
    return js_div

def get_layer_output(model, moe_layer_idx, tokenizer, input_strs, batch_size=1, add_special_tokens=True):
    model = model.eval()
    layer_idx = moe_layer_idx + 2  # add embedding layer and ffn layer

    def encode_text_batch(input_strs):
        inputs = tokenizer.batch_encode_plus(
            input_strs,
            padding='longest',
            add_special_tokens=add_special_tokens,
            return_tensors="pt"
        )
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        return input_ids, attention_mask

    num_texts = len(input_strs)
    layer_outputs = []

    for i in range(0, num_texts, batch_size):
        text_list_batch = input_strs[i:i + batch_size]
        input_ids, attention_mask = encode_text_batch(text_list_batch)
        with torch.no_grad():
            outputs = model(
                input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            layer_output = hidden_states[layer_idx]
            layer_output = layer_output.to(torch.float32)

            # Remove padding based on attention mask
            for j in range(len(text_list_batch)):
                # print(layer_output[j].size())
                length = attention_mask[j].sum().item()  # the valid length of the input
                trimmed_output = layer_output[j, :length, :]
                # print(trimmed_output.size())
                layer_outputs.append(trimmed_output)

    return layer_outputs

def get_total_js_divergence(origin_layer_outputs, prune_layer_outputs):
    js_div_sum = 0.0
    for o, p in zip(origin_layer_outputs, prune_layer_outputs):
        js_div = calculate_js_divergence(o, p)
        js_div_sum += js_div.item()
    mean_js_div = js_div_sum / len(origin_layer_outputs)
    print("sum div {} length dataset {} mean div {}".format(
        js_div_sum, len(origin_layer_outputs), mean_js_div))
    return mean_js_div


parser = argparse.ArgumentParser()
parser.add_argument("--input", default="./datasets/sample_eval.json",
                    help="eval数据集路径")
parser.add_argument("--model", default="./deepseek",
                    help="模型路径")
parser.add_argument("--batch-size", type=int, default=8, help="并行解码的样本数量")
parser.add_argument("--num-layer", type=int, default=27,
                    help="默认为qw16B层数")  # deepseek 27 qw24
parser.add_argument("--num-expert", type=int, default=64, help="默认为qw16B专家数")

parser.add_argument("--prune-layer", default=0, type=int,
                    help="选定一层进行剪枝")
parser.add_argument("--reverse-experts", action="store_true",
                    help="如果指定，则剪枝时倒转expert顺序")

args = parser.parse_args()

pytorch_checkpoint_path = args.model
# @param ["", "0", "0,1", "0,1,2"] {allow-input: true}
available_gpu_ids_str = "0"
memory_per_gpu = "38GiB"  # @param ["", "38GiB"] {allow-input: true}
cpu_memory = '50GiB'  # @param ["50GiB"] {allow-input: true}
model_dtype = 'bfloat16'  # @param ["float32", "bfloat16"]
offload = False  # @param {type:"boolean"}

if torch.cuda.is_available():
    cuda_list = available_gpu_ids_str.split(',')
else:
    available_gpu_ids_str, memory_per_gpu = "", ""
    model_dtype = "bfloat16"
    cuda_list = []

no_split_module_classes = "OpenMoeDecoderLayer"

# 1. Allocate Devices for Inference
available_memory = {int(cuda): memory_per_gpu for cuda in cuda_list}
available_memory['cpu'] = cpu_memory
print('Available Devices and Memory: ', available_memory)

# 2. Load the Model (init with empty weight to save memory)
config = AutoConfig.from_pretrained(
    pytorch_checkpoint_path, trust_remote_code=True)
# weights_location = snapshot_download(repo_id=pytorch_checkpoint_path)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config,
                                             torch_dtype=eval(
                                                 f'torch.{model_dtype}'),
                                             trust_remote_code=True)
print('Model dtype: ', model.dtype)
device_map = infer_auto_device_map(model,
                                   max_memory=available_memory,
                                   no_split_module_classes=no_split_module_classes)
print('Inferred Device Map: \n', device_map)


model = AutoModelForCausalLM.from_pretrained(
    pytorch_checkpoint_path,
    device_map=device_map,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    # offload_folder="offload",
    # offload_state_dict=True,
    # dtype=eval(f'torch.{model_dtype}'),
    # no_split_module_classes=[no_split_module_classes]
)
tokenizer = AutoTokenizer.from_pretrained(pytorch_checkpoint_path)

# read benchmark
# with open(args.input, 'r') as fp:
#     questions = []
#     for line in fp:
#         line = line.strip()
#         if line:
#             question = json.loads(line)
#             questions.append(question)
# raw_questions = list(map(lambda x: x["turns"][0], questions))
with open(args.input, 'r') as fp:
    questions = []
    for line in fp:
        line = line.strip()
        if line:
            question = json.loads(line)
            questions.append(question)
raw_questions = list(map(lambda x: x["text"], questions))


batch_size = args.batch_size
num_layer = args.num_layer
num_expert = args.num_expert
# prune_num_expert = args.prune_num_expert

# load dynamic weights
dynamic_weight_tmp = json.load(open("deepseek_model/dynamic_weight.json"))
for key, value in dynamic_weight_tmp.items():
    key = key.split("-")
    layer_idx = int(key[0])
    expert_idx = int(key[1])
    w = value[-1]
    dynamic_weights[(layer_idx, expert_idx)] = w
print(dynamic_weights)


# origin output
prune_layer_list.append({})
layer_num_list.append(num_layer)
import time
s = time.time()
origin_get_layer_output = get_layer_output(model, 0, tokenizer, raw_questions, batch_size=batch_size)
e = time.time()
print("compute origin layer output cost {}".format(e-s))

# prune
prune_layer_idx = int(args.prune_layer)  # 每次只剪枝一层，逐层看效果
prune_expert_idx_list = []  # greedy search expert list
output_dict = {"expert_idxs": [],
               "mean_jl": [],
               "expert_num": []}
try:
    while (len(prune_expert_idx_list) < 6):  # extra expert num = 6
        print("the {}th iteration".format(len(prune_expert_idx_list)))
        candidate_expert_idx_list = [expert for expert in range(64)
                                     if expert not in prune_expert_idx_list]
        # candidate_layer_idx_list = candidate_layer_idx_list[:beam_size]
        print("exist prune experts {}; candidate prune experts {}".format(
            prune_expert_idx_list, candidate_expert_idx_list))

        optimal_jl = 1000000
        optimal_candidate_idx = -1
        for candidate_idx in candidate_expert_idx_list:  # greedy search expert
            start_time = time.time()
            tmp_prune_expert_idx_list = prune_expert_idx_list + \
                [candidate_idx]  # 确定layer
            print("try to eval expert idx list {}".format(
                tmp_prune_expert_idx_list))

            prune_layer_idx_to_expert_idxs = {prune_layer_idx: tmp_prune_expert_idx_list}
            print("prune layer idx to expert idxs {}".format(
                prune_layer_idx_to_expert_idxs))
            # update prune variables
            prune_layer_list.append(prune_layer_idx_to_expert_idxs)
            layer_num_list.append(num_layer)

            # eval ppl on benchmark
            prune_get_layer_output = get_layer_output(model, 0, tokenizer, raw_questions, batch_size=batch_size)
            mean_jl = get_total_js_divergence(origin_get_layer_output, prune_get_layer_output)

            output_dict["mean_jl"].append(mean_jl)
            output_dict["expert_idxs"].append(tmp_prune_expert_idx_list)
            output_dict["expert_num"].append(len(tmp_prune_expert_idx_list))

            if mean_jl < optimal_jl:
                optimal_jl = mean_jl
                optimal_candidate_idx = candidate_idx

            end_time = time.time()
            print("jl {}, best_jl {}, eval jl cost {} seconds\n".format(
                mean_jl, optimal_jl, end_time-start_time))

        prune_expert_idx_list = prune_expert_idx_list + [optimal_candidate_idx]

        output_dict["mean_jl"].append(optimal_jl)
        output_dict["expert_idxs"].append(prune_expert_idx_list)
        output_dict["expert_num"].append(len(prune_expert_idx_list))

    print(output_dict)
    output_df = pd.DataFrame(output_dict)
    output_df.to_excel(
        "greedy_search_expert_output_by_jl/greedy_search_expert_jl_layer{}.xlsx".format(prune_layer_idx))
except Exception as e:
    import traceback
    msg = traceback.format_exc()
    print("error: {}, {}".format(e, msg))
