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



# prune
prune_layer_idx = int(args.prune_layer) # 每次只剪枝一层，逐层看效果
prune_expert_idx_list = []  # greedy search expert list
output_dict = {"expert_idxs": [],
               "ppl": [],
               "expert_num": []}
try:
    while (len(prune_expert_idx_list) < 6):
        print("the {}th iteration".format(len(prune_expert_idx_list)))
        candidate_expert_idx_list = [expert for expert in range(64)
                                    if expert not in prune_expert_idx_list]
        # candidate_layer_idx_list = candidate_layer_idx_list[:beam_size]
        print("exist prune experts {}; candidate prune experts {}".format(
            prune_expert_idx_list, candidate_expert_idx_list))

        optimal_ppl = 1000000
        optimal_candidate_idx = -1
        for candidate_idx in candidate_expert_idx_list:
            start_time = time.time()
            tmp_prune_expert_idx_list = prune_expert_idx_list + \
                [candidate_idx]  # 确定layer
            print("try to eval expert idx list {}".format(tmp_prune_expert_idx_list))

            prune_layer_idx_to_expert_idxs = {0: tmp_prune_expert_idx_list}
            print("prune layer idx to expert idxs {}".format(
                prune_layer_idx_to_expert_idxs))
            # update prune variables
            prune_layer_list.append(prune_layer_idx_to_expert_idxs)
            layer_num_list.append(num_layer)

            # eval ppl on benchmark
            mean_ppl = compute_ppl(model, tokenizer, raw_questions, None)
            output_dict["ppl"].append(mean_ppl)
            output_dict["expert_idxs"].append(tmp_prune_expert_idx_list)
            output_dict["expert_num"].append(len(tmp_prune_expert_idx_list))

            if mean_ppl < optimal_ppl:
                optimal_ppl = mean_ppl
                optimal_candidate_idx = candidate_idx

            end_time = time.time()
            print("ppl {}, best_ppl {}, eval ppl cost {} seconds".format(mean_ppl, optimal_ppl, end_time-start_time))

        prune_expert_idx_list = prune_expert_idx_list + [optimal_candidate_idx]

    print(output_dict)
    output_df = pd.DataFrame(output_dict)
    output_df.to_excel("greedy_search_expert_layer{}.xlsx".format(prune_layer_idx))
except Exception as e:
    import traceback
    msg = traceback.format_exc()
    print("error: {}, {}".format(e, msg))
