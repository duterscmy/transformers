# -*- coding: utf-8 -*-
from peft import LoraConfig, get_peft_model
import shutil
import traceback
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from transformers import AutoTokenizer
from datasets import load_dataset
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


parser = argparse.ArgumentParser()
parser.add_argument("--input", default="datasets/c4-train.00000-of-01024.head1w.json",
                    help="finetune data")
parser.add_argument("--model", default="./deepseek",
                    help="模型路径")
parser.add_argument("--batch-size", type=int, default=8, help="并行解码的样本数量")
parser.add_argument("--num-layer", type=int, default=27,
                    help="默认为qw16B层数")  # deepseek 27 qw24
parser.add_argument("--num-expert", type=int, default=64, help="默认为qw16B专家数")


parser.add_argument("--score-mode", type=str, default="l1", help="层间对专家排序的指标")
parser.add_argument("--prune-num-expert", default=0, type=int,
                    help="剪枝后剩余的expert数量")
parser.add_argument("--prune-num-layer", default=9, type=int,
                    help="剪枝后剩余的layer数量")
parser.add_argument("--reverse-experts", action="store_true",
                    help="如果指定，则剪枝时倒转expert顺序")

args = parser.parse_args()

pytorch_checkpoint_path = args.model
# @param ["", "0", "0,1", "0,1,2"] {allow-input: true}
available_gpu_ids_str = "0"
memory_per_gpu = "48GiB"  # @param ["", "38GiB"] {allow-input: true}
cpu_memory = '80GiB'  # @param ["50GiB"] {allow-input: true}
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
device_map = {'': 'cuda:0'}


model = AutoModelForCausalLM.from_pretrained(
    pytorch_checkpoint_path,
    device_map=device_map,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(pytorch_checkpoint_path)


batch_size = args.batch_size
score_mode = args.score_mode
num_layer = args.num_layer
num_expert = args.num_expert
prune_num_expert = args.prune_num_expert
prune_num_layer = args.prune_num_layer

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
    layer_idx_to_expert_idxs = json.load(
        open("deepseek_model/layer_idx_to_expert_idx.random.json", 'r'))
    layer_idx_to_expert_idxs = {
        int(key): value for key, value in layer_idx_to_expert_idxs.items()}
elif score_mode == "greedy_jl":
    layer_idx_to_expert_idxs = json.load(
        open("deepseek_model/layer_idx_to_expert_idx.greedy_jl.json", 'r'))
    layer_idx_to_expert_idxs = {
        int(key): value for key, value in layer_idx_to_expert_idxs.items()}


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
if prune_num_expert == 0:
    # layer_idx_list_ppl_order = [11, 18, 7, 8, 2, 23, 10, 22, 13, 16,
    #                             15, 20, 24, 19, 25, 4, 6, 5, 3, 9, 21, 27, 17, 12, 26, 14, 1]
    # layer_idx_list_ppl_order = [layer-1 for layer in layer_idx_list_ppl_order]
    layer_idx_list_ppl_order = [10, 17, 22, 1, 12,
                                21, 6, 15, 7, 19, 24, 9]
elif prune_num_expert == 6 and score_mode == "random":
    # layer_idx_list_ppl_order = [11, 18, 7, 23, 15, 8, 10, 2, 22, 20,
    #                             24, 16, 13, 6, 3, 19, 25, 4, 5, 9, 21, 27, 17, 12, 26, 14, 1]
    # layer_idx_list_ppl_order = [layer-1 for layer in layer_idx_list_ppl_order]
    layer_idx_list_ppl_order = [10, 17, 22, 1, 9,
                                6, 21, 15, 12, 14, 19, 7]
elif prune_num_expert == 6 and score_mode == "l1":
    layer_idx_list_ppl_order = [5, 18, 11, 22, 8, 13, 10, 7, 23, 16,
                                2, 20, 4, 24, 15, 19, 9, 3, 25, 6, 17, 1, 21, 27, 14, 12, 26]
    layer_idx_list_ppl_order = [layer-1 for layer in layer_idx_list_ppl_order]
    layer_idx_list_ppl_order = [4, 17, 21, 10, 22,
                                12, 7, 15, 9, 19, 6, 18]
elif prune_num_expert == 6 and score_mode == "distribution":
    layer_idx_list_ppl_order = [15, 10, 7, 18, 8, 2, 22, 16, 23, 11,
                                20, 24, 13, 6, 19, 25, 4, 3, 5, 1, 27, 9, 21, 17, 12, 26, 14]
    layer_idx_list_ppl_order = [layer-1 for layer in layer_idx_list_ppl_order]
    layer_idx_list_ppl_order = [14, 9, 1, 21, 22,
                                6, 17, 10, 12, 7, 15, 24]
elif prune_num_expert == 6 and score_mode == "greedy_jl":
    layer_idx_list_ppl_order = [19, 15, 22, 10, 12, 6, 14, 21, 26, 7, 17, 1, 24, 23, 9]


# add expert weight to prune layer
for param in model.parameters():
    param.requires_grad = False


prune_layer_idx_to_expert_idx = {}
for prune_layer_idx in layer_idx_list_ppl_order[:prune_num_layer]:
    prune_expert_idx_list = layer_idx_to_expert_idxs[prune_layer_idx][:prune_num_expert]
    prune_layer_idx_to_expert_idx[prune_layer_idx] = prune_expert_idx_list
# set global variable
prune_layer_list.append(prune_layer_idx_to_expert_idx)
layer_num_list.append(num_layer)


# add lora to model
def check_if_lora(module_name):
    try:
        layer_id = int(module_name.split(".")[2])
        expert_id = int(module_name.split(".")[5])
    except:
        return False
    moe_layer_id = layer_id - 1
    # print(moe_layer_id, expert_id)
    if moe_layer_id in prune_layer_idx_to_expert_idx and expert_id in prune_layer_idx_to_expert_idx[moe_layer_id]:
        return True
    return False


linear_module_list = []
for name, module in model.named_modules():
    if isinstance(module, (torch.nn.Linear)):
        linear_module_list.append(name)
finetune_module_list = list(filter(check_if_lora, linear_module_list))
print("finetune_module_list: {}".format(finetune_module_list))

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

print_trainable_parameters(model)
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=finetune_module_list,
    lora_dropout=0.01,
    bias="none"
    # task_type="SEQ_2_SEQ_LM",
)
lora_model = get_peft_model(model, config)
print_trainable_parameters(lora_model)

# exit()
# finetune
# 加载数据集
dataset = load_dataset('json', data_files=[
                       args.input])
eval_dataset = load_dataset('json', data_files=["datasets/sample_questions_from_6_dataset.json"])


# 假设你正在使用GPT-2模型（你可以根据需要更改为其他模型）
tokenizer = AutoTokenizer.from_pretrained(pytorch_checkpoint_path)
# 定义tokenize函数


def tokenize_function(examples):
    x = tokenizer(examples['text'], padding="max_length",
                  truncation=True, max_length=256, return_tensors="pt")
    # Ensure labels are the same as input_ids and convert to FP32
    x["labels"] = x["input_ids"].clone()
    return x


# 应用tokenize函数
tokenized_datasets = dataset.map(
    tokenize_function, batched=True, remove_columns=["text"])

eval_tokenized_datasets = eval_dataset.map(
    tokenize_function, batched=True, remove_columns=["text"])
# tokenized_datasets = tokenized_datasets.map(add_labels, batched=True)
# 设置格式化输出
# tokenized_datasets.set_format(
#     type='torch', columns=['input_ids', 'attention_mask'])

# 设置训练参数

output_file = "finetune_lora_score_mode_{}_layer_{}.json".format(
        score_mode, prune_num_layer)
output_dir = "deepseek_model/finetune_lora"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

output_path = os.path.join(output_dir, output_file)
training_args = TrainingArguments(
    output_dir=output_path,          # 输出文件夹（注意：尽管设置了output_dir，但模型不会被保存）
    overwrite_output_dir=True,               # 覆盖输出文件夹
    num_train_epochs=1,                      # 训练轮数
    per_device_train_batch_size=args.batch_size,           # 每个设备的batch大小
    save_steps=200,                         # 不保存检查点（或者设置一个非常大的值，如1000000）
    save_strategy="steps",
    save_total_limit=0,                      # 不保存任何检查点（虽然设置为0在某些情况下可能不是必需的，但这里为了明确性）
    logging_steps=5,                        # 日志记录的步数
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    eval_steps=100,                         # 不保存检查点（或者设置一个非常大的值，如1000000）
    eval_strategy="steps",
    # 注意：其他参数可以根据需要进行调整
)
# 初始化Trainer
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=eval_tokenized_datasets["train"],
)
trainer.train()

