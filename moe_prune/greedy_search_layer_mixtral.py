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
    # print("probs p {}, probs q {}".format(probs_p, probs_q))
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
    # print("p {}, q {}".format(p, q))
    m = 0.5 * (p + q)
    kl_pm = calculate_kl_divergence(p, m)
    kl_qm = calculate_kl_divergence(q, m)
    js_div = 0.5 * (kl_pm + kl_qm)
    # print("kl per sample {} {} {}".format(kl_pm, kl_qm, js_div))
    return js_div


def get_layer_output(model, moe_layer_idx, tokenizer, input_strs, batch_size=1, add_special_tokens=True):
    model = model.eval()
    layer_idx = moe_layer_idx + 1  # add embedding layer

    def encode_text_batch(input_strs):
        inputs = tokenizer.batch_encode_plus(
            input_strs,
            padding='longest',
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        # print("input_ids {}".format(input_ids))
        # print("attention mask {}".format(attention_mask))
        return input_ids, attention_mask

    num_texts = len(input_strs)
    layer_outputs = []

    for i in range(0, num_texts, batch_size):
        text_list_batch = input_strs[i:i + batch_size]
        input_ids, attention_mask = encode_text_batch(text_list_batch)
        print(input_ids.size())
        with torch.no_grad():
            outputs = model(
                input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            layer_output = hidden_states[layer_idx]
            layer_output = layer_output.to(torch.float32)
            # print("layer output {}".format(layer_output))
            # print("layer output size {}".format(layer_output.size()))
            # Remove padding based on attention mask
            for j in range(len(text_list_batch)):
                # print(layer_output[j].size())
                length = attention_mask[j].sum().item()  # the valid length of the input
                trimmed_output = layer_output[j, -length:, :]  # 左侧padding
                # print("trimmed output {}".format(trimmed_output))
                # print("trimeed output dtype {}".format(trimmed_output.dtype))
                layer_outputs.append(trimmed_output)
    return layer_outputs

def get_total_js_divergence(origin_layer_outputs, prune_layer_outputs):
    js_div_sum = 0.0
    for o, p in zip(origin_layer_outputs, prune_layer_outputs):
        js_div = calculate_js_divergence(o, p)
        js_div_sum += js_div.item()
    mean_js_div = js_div_sum / len(origin_layer_outputs)
    print("sum div {}, length dataset {}, mean div {}".format(
        js_div_sum, len(origin_layer_outputs), mean_js_div), flush=True)
    return mean_js_div


parser = argparse.ArgumentParser()
parser.add_argument("--input", default="./datasets/sample_eval.json",
                    help="eval数据集路径")
parser.add_argument("--model", default="./deepseek",
                    help="模型路径")
parser.add_argument("--batch-size", type=int, default=8, help="并行解码的样本数量")
parser.add_argument("--num-layer", type=int, default=32)  # deepseek 27 qw24
parser.add_argument("--num-expert", type=int, default=8)

parser.add_argument("--prune-layer", default=16, type=int,
                    help="剪枝层的数量")
parser.add_argument("--prune-expert", default=2, type=int,
                    help="剪枝专家的数量")
parser.add_argument("--load-in-8bit", action="store_true", help="load in 8 bit")

args = parser.parse_args()

pytorch_checkpoint_path = args.model
no_split_module_classes = "OpenMoeDecoderLayer"

# 2. Load the Model (init with empty weight to save memory)
config = AutoConfig.from_pretrained(
    pytorch_checkpoint_path, trust_remote_code=True)

if args.load_in_8bit:
    model = AutoModelForCausalLM.from_pretrained(
        pytorch_checkpoint_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        load_in_8bit = True
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        pytorch_checkpoint_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

tokenizer = AutoTokenizer.from_pretrained(pytorch_checkpoint_path)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

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
num_prune_expert = args.prune_expert
num_prune_layer = args.prune_layer

# load dynamic weights
if num_prune_expert == 0:
    layer_idx_to_expert_idxs = {idx: [] for idx in range(27)}
else:
    layer_idx_to_expert_idxs = json.load(
        open("mixtral/layer_idx_to_expert_idx.json", 'r'))
    layer_idx_to_expert_idxs = {
        int(key): value[:num_prune_expert] for key, value in layer_idx_to_expert_idxs.items()}

dynamic_weight_tmp = json.load(open("mixtral/dynamic_weights.mixtral.json"))
for key, value in dynamic_weight_tmp.items():
    key = key.split("-")
    layer_idx = int(key[0])
    expert_idx = int(key[1])
    w = value[-1]
    dynamic_weights[(layer_idx, expert_idx)] = w


# origin output (no prune)
prune_layer_idx = int(args.prune_layer)  # 每次只剪枝一层，逐层看效果
prune_layer_list.append({})
layer_num_list.append(num_layer)
s = time.time()
origin_get_layer_output = get_layer_output(
    model, num_layer-1, tokenizer, raw_questions, batch_size=batch_size)
e = time.time()
print("compute origin layer output cost {}".format(e-s))

# prune

beam_size = 1
max_greedy_layer_num = num_prune_layer
beam_prune_layer_idx_list = [[12, 14, 13, 8, 7, 20, 23, 22, 6, 16, 9, 25, 5, 24, 18]]
# prune_layer_idx_list = [19, 15, 22, 10, 12, 6, 14, 21,]  # greedy search expert list
# no_prune_list = [0, 8, 11, 13, 16, 20, 25 ,26]
output_dict = {"layer_idxs": [],
               "mean_jl": [],
               "layer_num": []}

while (len(beam_prune_layer_idx_list[0]) < max_greedy_layer_num):
    print("the {}th iteration".format(
        len(beam_prune_layer_idx_list[0])), flush=True)
    new_prune_layer_idx_list_with_jl = []

    for prune_layer_idx_list in beam_prune_layer_idx_list:
        candidate_layer_idx_list = [layer for layer in range(num_layer)
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

            # eval ppl on benchmark
            prune_get_layer_output = get_layer_output(
                model, num_layer-1, tokenizer, raw_questions, batch_size=batch_size)
            mean_jl = get_total_js_divergence(
                origin_get_layer_output, prune_get_layer_output)
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
    "mixtral/greedy_search_layer_prune_expert{}_beam{}.xlsx".format(num_prune_expert, beam_size))
