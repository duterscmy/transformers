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


def compute_ppl(model, tokenizer, input_strs, gen_kwargs=None,
                add_special_tokens=True, split_special_tokens=False, output_only=True, verbose=False):

    model = model.eval()

    # Tokenization
    def encode_text_batch(input_strs):
        inputs = tokenizer.batch_encode_plus(
            input_strs,
            padding='longest',
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
            max_length=256,
            truncation=True
        )
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        # print("input_ids {}".format(input_ids))
        # print("attention mask {}".format(attention_mask))
        return input_ids

    batch_size = 8  # 批处理大小
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
parser.add_argument("--input", default="./datasets/sample_eval.json",
                    help="eval数据集路径")
parser.add_argument("--model", default="./deepseek",
                    help="模型路径")
parser.add_argument("--dynamic-weight-file", default="./",
                    help="动态路由系数")
parser.add_argument("--greedy-expert-file",  default="./",
                    help="逐层贪心搜索的专家")
parser.add_argument("--batch-size", type=int, default=8, help="并行解码的样本数量")
parser.add_argument("--num-layer", type=int, default=27,
                    help="默认为qw16B层数")  # deepseek 27 qw24
parser.add_argument("--num-expert", type=int, default=64, help="默认为qw16B专家数")

parser.add_argument("--prune-layer", default=0, type=int,
                    help="剪枝层的数量")
parser.add_argument("--prune-expert", default=6, type=int,
                    help="剪枝专家的数量")
parser.add_argument("--prune-expert-strategy", default="greedy_jl", type=str,
                    help="剪枝专家的策略")
parser.add_argument("--reverse-experts", action="store_true",
                    help="如果指定，则剪枝时倒转expert顺序")

args = parser.parse_args()
batch_size = args.batch_size
num_layer = args.num_layer
num_expert = args.num_expert
num_prune_expert = args.prune_expert
prune_expert_strategy = args.prune_expert_strategy
pytorch_checkpoint_path = args.model
dynamic_weight_file = args.dynamic_weight_file
greedy_expert_file = args.greedy_expert_file

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

# load dynamic weights
if num_prune_expert == 0:
    layer_idx_to_expert_idxs = {idx: [] for idx in range(27)}
elif prune_expert_strategy == "greedy_jl":
    layer_idx_to_expert_idxs = json.load(
        open("deepseek_model/layer_idx_to_expert_idx.greedy_jl.json", 'r'))
    layer_idx_to_expert_idxs = {
        int(key): value[:num_prune_expert] for key, value in layer_idx_to_expert_idxs.items()}
elif prune_expert_strategy == "greedy_jl_c4":
    layer_idx_to_expert_idxs = json.load(
        open("deepseek_model/layer_idx_to_expert_idx.greedy_jl.c4.json", 'r'))
    layer_idx_to_expert_idxs = {
        int(key): value[:num_prune_expert] for key, value in layer_idx_to_expert_idxs.items()}

if prune_expert_strategy == "greedy_jl":
    dynamic_weight_file = "deepseek_model/dynamic_weight.json"
elif prune_expert_strategy == "greedy_jl_c4":
    dynamic_weight_file = "deepseek_model/dynamic_weight.c4.json"

dynamic_weight_tmp = json.load(open(dynamic_weight_file, 'r'))
for key, value in dynamic_weight_tmp.items():
    key = key.split("-")
    layer_idx = int(key[0])
    expert_idx = int(key[1])
    w = value[-1]
    dynamic_weights[(layer_idx, expert_idx)] = w
# print(dynamic_weights)

# prune

beam_size = 1
max_greedy_layer_num = 15
beam_prune_layer_idx_list = [[]]

output_dict = {"layer_idxs": [],
               "mean_jl": [],
               "layer_num": []}
try:
    while (len(beam_prune_layer_idx_list[0]) < max_greedy_layer_num):
        print("the {}th iteration".format(
            len(beam_prune_layer_idx_list[0])), flush=True)
        new_prune_layer_idx_list_with_jl = []

        for prune_layer_idx_list in beam_prune_layer_idx_list:
            candidate_layer_idx_list = [layer for layer in range(27)
                                        if layer not in prune_layer_idx_list]
            # candidate_layer_idx_list = candidate_layer_idx_list[:beam_size]
            print("exist prune layers {}; candidate prune layers {}".format(
                prune_layer_idx_list, candidate_layer_idx_list), flush=True)

            for candidate_idx in candidate_layer_idx_list:  # greedy search expert
                start_time = time.time()
                tmp_layer_list = prune_layer_idx_list + [candidate_idx]
                print("try to eval layer idx list {}".format(
                    tmp_layer_list), flush=True)

                prune_layer_idx_to_expert_idxs = {}
                for layer_idx in tmp_layer_list:
                    prune_layer_idx_to_expert_idxs[layer_idx] = layer_idx_to_expert_idxs[layer_idx]
                print("exp prune layer idx to expert idxs {}".format(
                    prune_layer_idx_to_expert_idxs), flush=True)
                # update prune variables
                prune_layer_list.append(prune_layer_idx_to_expert_idxs)
                layer_num_list.append(num_layer)

                mean_jl = compute_ppl(model, tokenizer, raw_questions)

                output_dict["mean_jl"].append(mean_jl)
                output_dict["layer_idxs"].append(tmp_layer_list)
                output_dict["layer_num"].append(len(tmp_layer_list))

                new_prune_layer_idx_list_with_jl.append(
                    (tuple(tmp_layer_list), mean_jl))

        new_prune_layer_idx_list_with_jl = sorted(
            new_prune_layer_idx_list_with_jl, key=lambda x: x[1])
        new_prune_layer_idx_list_with_jl = new_prune_layer_idx_list_with_jl[:beam_size]
        for prune_layer_idx_tuple, jl in new_prune_layer_idx_list_with_jl:
            output_dict["mean_jl"].append(jl)
            output_dict["layer_idxs"].append(prune_layer_idx_tuple)
            output_dict["layer_num"].append(len(prune_layer_idx_tuple))
        beam_prune_layer_idx_list = [
            list(t) for t, j in new_prune_layer_idx_list_with_jl]

    output_df = pd.DataFrame(output_dict)
    output_df.to_excel(
        "greedy_search_layer_prune_expert{}_stratedy{}_ppl_beam{}.xlsx".format(num_prune_expert, prune_expert_strategy, beam_size))
except Exception as e:
    import traceback
    msg = traceback.format_exc()
    print("error: {}, {}".format(e, msg), flush=True)
