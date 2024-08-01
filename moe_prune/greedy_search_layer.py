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


def compute_ppl(model, tokenizer, input_strs, gen_kwargs,
                add_special_tokens=True, split_special_tokens=False, output_only=True, verbose=False):

    model = model.eval()

    # Tokenization
    def encode_text_batch(input_strs):
        inputs = tokenizer.batch_encode_plus(input_strs,
                                             padding='longest',
                                             #  add_special_tokens=add_special_tokens,
                                             #  split_special_tokens=split_special_tokens,
                                             return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        return input_ids

    batch_size = 1  # 批处理大小
    num_texts = len(input_strs)
    loss_sum = 0.0

    for i in range(0, len(input_strs), batch_size):
        text_list_batch = input_strs[i:i+batch_size]
        input_ids = encode_text_batch(text_list_batch)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss.mean()
            # print("mean loss {}".format(loss))
        loss_sum += loss.item()
        # print("loss sum {}".format(loss_sum))

    mean_loss = loss_sum / num_texts  # 计算整个数据集的损失均值
    mean_ppl = torch.exp(torch.tensor(mean_loss))
    return mean_ppl


parser = argparse.ArgumentParser()
parser.add_argument("--input", default="./moe_prune/data/questions.jsonl",
                    help="MTBench数据集路径")
parser.add_argument("--model", default="./deepseek",
                    help="模型路径")
parser.add_argument("--batch-size", type=int, default=4, help="并行解码的样本数量")
parser.add_argument("--num-layer", type=int, default=27,
                    help="默认为qw16B层数")  # deepseek 27 qw24
parser.add_argument("--num-expert", type=int, default=64, help="默认为qw16B专家数")


parser.add_argument("--score-mode", type=str, default="l1", help="层间对专家排序的指标")
parser.add_argument("--prune-num-expert", default=0, type=int,
                    help="剪枝后剩余的expert数量")
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
if "qw27" in pytorch_checkpoint_path:
    for layer in model.model.layers:
        layer.mlp.split()


# read benchmark
with open(args.input, 'r') as fp:
    questions = []
    for line in fp:
        line = line.strip()
        if line:
            question = json.loads(line)
            questions.append(question)
raw_questions = list(map(lambda x: x["turns"][0], questions))


batch_size = args.batch_size
score_mode = args.score_mode
num_layer = args.num_layer
num_expert = args.num_expert
prune_num_expert = args.prune_num_expert


# prune layer idx and expert idx
if score_mode == "l1":
    layer_idx_to_expert_idxs = json.load(
        open("deepseek_model/layer_idx_to_expert_idx.json", 'r'))
    layer_idx_to_expert_idxs = {
        int(key): value for key, value in layer_idx_to_expert_idxs.items()}
elif score_mode == "ww_alpha":
    layer_idx_to_expert_idxs = json.load(
        open("deepseek_model/layer_idx_to_expert_idx.alpha.json", 'r'))
    layer_idx_to_expert_idxs = {
        int(key): value for key, value in layer_idx_to_expert_idxs.items()}
elif score_mode == "distribution":
    layer_idx_to_expert_idxs = json.load(
        open("deepseek_model/layer_idx_to_expert_idx.distribution.json", 'r'))
    layer_idx_to_expert_idxs = {
        int(key): value for key, value in layer_idx_to_expert_idxs.items()}
elif score_mode == "random":
    # layer_idx_to_expert_idxs = {}
    # for layer_idx in range(num_layer):
    #     expert_idxs = list(range(num_expert))
    #     random.shuffle(expert_idxs)
    #     layer_idx_to_expert_idxs[layer_idx] = expert_idxs
    layer_idx_to_expert_idxs = json.load(
        open("deepseek_model/layer_idx_to_expert_idx.random.json", 'r'))
    layer_idx_to_expert_idxs = {
        int(key): value for key, value in layer_idx_to_expert_idxs.items()}


# decode and eval ppl
# no prune
# mean_ppl = compute_ppl(model, tokenizer, raw_questions, None)
# print("no prune mean_ppl {}".format(mean_ppl))
# mean_ppl = mean_ppl.tolist()
# output = {"mean_ppl": mean_ppl}
# model_id = "noPrune"
# output_filename = "{}.json".format(model_id)
# output_filename = os.path.join(output_path, output_filename)
# json.dump(output, open(output_filename, 'w'))

# load dynamic weights
dynamic_weight_tmp = json.load(open("deepseek_model/dynamic_weight.json"))
for key, value in dynamic_weight_tmp.items():
    key = key.split("-")
    layer_idx = int(key[0])
    expert_idx = int(key[1])
    w = value[-1]
    dynamic_weights[(layer_idx, expert_idx)] = w
print(dynamic_weights)

# ppl order pruning single layer
# ! exist 1 offset
if prune_num_expert == 0:
    layer_idx_list_ppl_order = [11, 18, 7, 8, 2, 23, 10, 22, 13, 16,
                                15, 20, 24, 19, 25, 4, 6, 5, 3, 9, 21, 27, 17, 12, 26, 14, 1]
elif prune_num_expert == 6 and score_mode == "random":
    layer_idx_list_ppl_order = [11, 18, 7, 23, 15, 8, 10, 2, 22, 20,
                                24, 16, 13, 6, 3, 19, 25, 4, 5, 9, 21, 27, 17, 12, 26, 14, 1]
elif prune_num_expert == 6 and score_mode == "l1":
    layer_idx_list_ppl_order = [5, 18, 11, 22, 8, 13, 10, 7, 23, 16,
                                2, 20, 4, 24, 15, 19, 9, 3, 25, 6, 17, 1, 21, 27, 14, 12, 26]
elif prune_num_expert == 6 and score_mode == "distribution":
    layer_idx_list_ppl_order = [15, 10, 7, 18, 8, 2, 22, 16, 23, 11,
                                20, 24, 13, 6, 19, 25, 4, 3, 5, 1, 27, 9, 21, 17, 12, 26, 14]
    
layer_idx_list_ppl_order = [layer-1 for layer in layer_idx_list_ppl_order]

# prune
prune_layer_idx_list = [layer_idx_list_ppl_order[0]]
beam_size = 5
output_dict = {"expert_idxs": [],
               "ppl": [],
               "expert_num": []}

while (len(prune_layer_idx_list) < 12):
    print("the {}th iteration".format(len(prune_layer_idx_list)))
    candidate_layer_idx_list = [layer for layer in layer_idx_list_ppl_order
                                if layer not in prune_layer_idx_list]
    candidate_layer_idx_list = candidate_layer_idx_list[:beam_size]
    print("exist prune layers {}; candidate prune layers {}".format(
        prune_layer_idx_list, candidate_layer_idx_list))

    optimal_ppl = 1000000
    optimal_candidate_idx = -1
    for candidate_idx in candidate_layer_idx_list:
        tmp_prune_layer_idx_list = prune_layer_idx_list + \
            [candidate_idx]  # 确定layer
        print("try to eval expert idx list {}".format(tmp_prune_layer_idx_list))

        prune_layer_idx_to_expert_idxs = {}  # 确定专家
        for prune_layer_idx in tmp_prune_layer_idx_list:
            true_prune_num = prune_num_expert
            prune_expert_idxs = layer_idx_to_expert_idxs[prune_layer_idx]
            prune_expert_idxs = list(map(int, prune_expert_idxs))
            if args.reverse_experts:
                prune_expert_idxs.reverse()
            prune_layer_idx_to_expert_idxs[prune_layer_idx] = prune_expert_idxs[:true_prune_num]

        print("prune layer idx to expert idxs {}".format(
            prune_layer_idx_to_expert_idxs))
        # update prune variables
        prune_layer_list.append(prune_layer_idx_to_expert_idxs)
        layer_num_list.append(num_layer)

        # eval ppl on benchmark
        mean_ppl = compute_ppl(model, tokenizer, raw_questions, None)
        output_dict["ppl"].append(mean_ppl)
        output_dict["expert_idxs"].append(tmp_prune_layer_idx_list)
        output_dict["expert_num"].append(len(tmp_prune_layer_idx_list))
        print("ppl {}".format(mean_ppl))
        if mean_ppl < optimal_ppl:
            optimal_ppl = mean_ppl
            optimal_candidate_idx = candidate_idx

    prune_layer_idx_list = prune_layer_idx_list + [optimal_candidate_idx]

print(output_dict)
output_df = pd.DataFrame(output_dict)
output_df.to_excel("greedy_search_layer_score_mode_{}_prune_expert_num_{}_beam_size_{}.xlsx".format(
    score_mode, prune_num_expert, beam_size))
