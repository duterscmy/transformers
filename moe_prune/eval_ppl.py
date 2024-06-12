# -*- coding: utf-8 -*-
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import torch
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoTokenizer, T5Tokenizer, AutoConfig, AutoModelForCausalLM, LogitsProcessorList, LogitsProcessor
from typing import List, Optional
import random
import os
import json
#import shortuuid
import time

from transformers.models.qwen2_moe.expert_idx import expert_idxs_list, global_layer_list, prune_layer_list, layer_num_list


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
            print("mean loss {}".format(loss))
        loss_sum += loss.item()
        print("loss sum {}".format(loss_sum))

    mean_loss = loss_sum / num_texts  # 计算整个数据集的损失均值
    mean_ppl = torch.exp(torch.tensor(mean_loss))
    return mean_ppl


def apply_llama_chat_template(tokenizer, input_strs, sys_prompt):
    # Use LLaMA's Chat Template(A bit diffrent from original one at the beginning part, we may correct it to the standard llama prompt template later)
    # input_strs = [('user_input', 'user'), ('AI_response', 'assistant'), ...]
    tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'system' %}{{ '<<SYS>>\\n' + message['content'] + '\\n<</SYS>>\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ ' '  + message['content'] + ' ' + eos_token }}{% endif %}{% endfor %}"
    system_prompt = {'content': sys_prompt, 'role': 'system'}
    chat = [system_prompt] + [{'content': input_str,
                               'role': role} for input_str, role in input_strs]
    input_str = tokenizer.apply_chat_template(chat,
                                              tokenize=False,
                                              add_generation_prompt=True)
    return input_str


pytorch_checkpoint_path = "Qwen/Qwen1.5-MoE-A2.7B"
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
config = AutoConfig.from_pretrained(pytorch_checkpoint_path)
#weights_location = snapshot_download(repo_id=pytorch_checkpoint_path)
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
    "Qwen/Qwen1.5-MoE-A2.7B",
    device_map=device_map,
    torch_dtype=torch.bfloat16,
    # offload_folder="offload",
    # offload_state_dict=True,
    # dtype=eval(f'torch.{model_dtype}'),
    # no_split_module_classes=[no_split_module_classes]
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B")


parser = argparse.ArgumentParser()
parser.add_argument("--input", default="./moe_prune/data/questions.jsonl",
                    help="MTBench数据集路径")
parser.add_argument("--score-mode", type=str, default="l1", help="层间对专家排序的指标")
parser.add_argument("--batch-size", type=int, default=4, help="并行解码的样本数量")
parser.add_argument("--prune-one-layer", action="store_true",
                    help="如果指定，则只剪枝一层，否则累加前面所有层")

args = parser.parse_args()


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
num_layer = 24
num_expert = 64

add_up = 1 if not args.prune_one_layer else 0
output_path = "qwen2.7b_score_{}_add_up_{}".format(score_mode, add_up)

print(f"{pytorch_checkpoint_path} num_layer {num_layer} num_expert {num_expert}")
if not os.path.exists(output_path):
    os.makedirs(output_path)

# prune layer idx and expert idx
if score_mode == "l1":
    layer_idx_to_expert_idxs = json.load(
        open("moe_prune/layer_idx_to_expert_idx.json", 'r'))
    layer_idx_to_expert_idxs = {int(key): value for key, value in layer_idx_to_expert_idxs.items()}
elif score_mode == "ww_alpha":
    layer_idx_to_expert_idxs = json.load(
        open("moe_prune/layer_idx_to_expert_idx.alpha.json", 'r'))
    layer_idx_to_expert_idxs = {int(key): value for key, value in layer_idx_to_expert_idxs.items()}
elif score_mode == "random":
    layer_idx_to_expert_idxs = {}
    for layer_idx in range(32):
        expert_idxs = list(range(64))
        random.shuffle(expert_idxs)
        layer_idx_to_expert_idxs[layer_idx] = expert_idxs


# decode and eval ppl
# no prune
mean_ppl = compute_ppl(model, tokenizer, raw_questions, None)
print("no prune mean_ppl {}".format(mean_ppl))
mean_ppl = mean_ppl.tolist()
output = {"mean_ppl": mean_ppl}
model_id = "noPrune"
output_filename = "{}.json".format(model_id)
output_filename = os.path.join(output_path, output_filename)
json.dump(output, open(output_filename, 'w'))

# prune
for prune_layer_num in range(1, num_layer+1):  # 对前多少层进行剪枝
    print("prune layer num {}".format(prune_layer_num))
    for prune_expert_num in [4, 8, 16]:  # 保留的专家数量
        print("prune expert num {}".format(prune_expert_num))
        prune_layer_idx_to_expert_idxs = {}
        prune_layer_idx_list = [
            prune_layer_num-1] if args.prune_one_layer else list(range(prune_layer_num))

        for prune_layer_idx in prune_layer_idx_list:
            if prune_expert_num == 4:
                prune_layer_idx_to_expert_idxs[prune_layer_idx] = []
            else:
                true_prune_num = prune_expert_num - 4
                prune_expert_idxs = layer_idx_to_expert_idxs[prune_layer_idx]
                prune_expert_idxs = list(map(int, prune_expert_idxs))
                prune_expert_idxs = list(
                    filter(lambda x: x not in (60, 61, 62, 63), prune_expert_idxs))
                prune_layer_idx_to_expert_idxs[prune_layer_idx] = prune_expert_idxs[:true_prune_num]

        print("prune layer idx to expert idxs {}".format(
            prune_layer_idx_to_expert_idxs))
        # update prune variables
        prune_layer_list.append(prune_layer_idx_to_expert_idxs)
        layer_num_list.append(num_layer)

        # eval ppl on benchmark
        mean_ppl = compute_ppl(model, tokenizer, raw_questions, None)
        print("mean_ppl {}".format(mean_ppl))
        mean_ppl = mean_ppl.tolist()
        output = {"mean_ppl": mean_ppl}
        # expert_idxs_str = expert_idxs
        model_id = "pruneLayerNum{}_pruneExpertNum{}".format(
            prune_layer_num, prune_expert_num)
        output_filename = "{}.json".format(model_id)
        output_filename = os.path.join(output_path, output_filename)
        json.dump(output, open(output_filename, 'w'))
