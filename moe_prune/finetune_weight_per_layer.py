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


# load dynamic weights
dynamic_weight_tmp = json.load(open("deepseek_model/dynamic_weight.json"))
for key, value in dynamic_weight_tmp.items():
    key = key.split("-")
    layer_idx = int(key[0])
    expert_idx = int(key[1])
    w = value[-1]
    dynamic_weights[(layer_idx, expert_idx)] = w
# print(dynamic_weights)


# add expert weight to prune layer
for param in model.parameters():
    param.requires_grad = False


prune_layer_idx_to_expert_idx = {}
for prune_layer_idx in [args.prune_num_layer]:
    prune_expert_idx_list = layer_idx_to_expert_idxs[prune_layer_idx][:prune_num_expert]
    prune_layer_idx_to_expert_idx[prune_layer_idx] = prune_expert_idx_list
# set global variable
prune_layer_list.append(prune_layer_idx_to_expert_idx)
print("prune layer and expert: {}".format(prune_layer_list))
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
# print("finetune_module_list: {}".format(finetune_module_list))


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
    lora_alpha=256,
    target_modules=finetune_module_list,
    lora_dropout=0.01,
    bias="none"
)
lora_model = get_peft_model(model, config)
print_trainable_parameters(lora_model)

# exit()
# finetune
# 加载数据集
dataset = load_dataset('json', data_files=[
                       args.input])


# 假设你正在使用GPT-2模型（你可以根据需要更改为其他模型）
tokenizer = AutoTokenizer.from_pretrained(pytorch_checkpoint_path)
# 定义tokenize函数


def tokenize_function(examples):
    x = tokenizer(examples['text'], padding="max_length",
                  truncation=True, max_length=512, return_tensors="pt")
    # Ensure labels are the same as input_ids and convert to FP32
    x["labels"] = x["input_ids"].clone()
    return x


# 应用tokenize函数
tokenized_datasets = dataset.map(
    tokenize_function, batched=True, remove_columns=["text"])

# 设置训练参数

output_file = "finetune_lora_score_mode_{}_layer_{}.json".format(
    score_mode, prune_num_layer)
output_dir = "deepseek_model/finetune_lora_per_layer"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

output_path = os.path.join(output_dir, output_file)
training_args = TrainingArguments(
    output_dir=output_path,          # 输出文件夹（注意：尽管设置了output_dir，但模型不会被保存）
    overwrite_output_dir=True,               # 覆盖输出文件夹
    num_train_epochs=1,                      # 训练轮数
    per_device_train_batch_size=args.batch_size,           # 每个设备的batch大小
    save_steps=500,                         # 不保存检查点（或者设置一个非常大的值，如1000000）
    save_strategy="steps",
    save_total_limit=0,                      # 不保存任何检查点（虽然设置为0在某些情况下可能不是必需的，但这里为了明确性）
    logging_steps=5,                        # 日志记录的步数
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1
    # 注意：其他参数可以根据需要进行调整
)
# 初始化Trainer
trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
)
trainer.train()


# eval ppl on test dataset
with open('./moe_prune/data/questions.jsonl', 'r') as fp:
    questions = []
    for line in fp:
        line = line.strip()
        if line:
            question = json.loads(line)
            questions.append(question)
raw_questions = list(map(lambda x: x["turns"][0], questions))

lora_model.eval()
mean_ppl = compute_ppl(lora_model, tokenizer, raw_questions, None)

print("mean_ppl on mtbench: {}".format(mean_ppl))
