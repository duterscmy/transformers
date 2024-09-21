from utils import print_trainable_parameters, \
    classify_shared_experts, \
    classify_remained_experts, \
    classify_pruned_experts

import gc
import argparse
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoConfig
import torch
import time
import subprocess
import os
os.environ['HF_DATASETS_CACHE'] = '/scratch-shared/'
os.environ['HF_TOKENIZERS_CACHE'] = '/scratch-shared/tokenizes'
os.environ['HF_HOME'] = '/scratch-shared/HF_HOME'
os.environ['HF_METRICS_CACHE'] = '/scratch-shared/metrics'
os.environ['HF_MODULES_CACHE'] = '/scratch-shared/modules'


# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Measure GPU memory and throughput of LLM inference")
parser.add_argument('--model_name', type=str, default="gpt2",
                    help="Name of the model to use")
parser.add_argument('--batch_size', type=int, default=1,
                    help="Batch size for inference")
parser.add_argument('--num_repeats', type=int, default=500,
                    help="Number of times to repeat the inference for averaging")

parser.add_argument("--prune-num-expert", default=6, type=int,
                    help="剪枝后剩余的expert数量")
parser.add_argument("--condense-num-layer", default=9, type=int,
                    help="剪枝后剩余的layer数量")
parser.add_argument("--layer-trim-num-layer", default=9, type=int,
                    help="layer trim层数")
args = parser.parse_args()

prune_num_expert = args.prune_num_expert
condense_num_layer = args.condense_num_layer
layer_trim_num_layer = args.layer_trim_num_layer

# Load a sample of the Wiki dataset
dataset = load_dataset("json",
                       data_files="datasets/c4-train.00000-of-01024.head2k.json",
                       trust_remote_code=True,
                       split='train')
print(dataset)

# Load the model and tokenizer
# ignore_mismatched_sizes=True
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
    ignore_mismatched_sizes=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    args.model_name, trust_remote_code=True)

for name,para in model.named_parameters():
    print(name)
exit()
for param in model.parameters():
    param.requires_grad = False

print(
    f"Average memory used during inference: {torch.cuda.memory_allocated()/1024**2} MB")
print_trainable_parameters(model)

# prune layer idx and expert idx
import json
layer_idx_to_expert_idxs = json.load(open('/root/transformers/qwen_model/layer_idx_to_expert_idx.greedy_jl.json', 'r'))
layer_idx_to_expert_idxs = {
    int(key): value for key, value in layer_idx_to_expert_idxs.items()}

if condense_num_expert == 4:
    condense_layer_order = [11, 9, 3, 13, 23, 18, 21, 20, 16, 10, 19, 22, 14, 4, 8]
else:
    condense_layer_order = [11, 3, 9, 14, 18, 23, 19, 12, 20, 8, 21, 16, 15, 10, 22]

layer_trim_layer_order = [20, 19, 18, 21, 11, 13, 17, 14, 15, 10, 9, 16, 22, 8,  4,  7,  5, 12,
         3,  6,  2,  1,  0, 23]
trim_layer_idxs = layer_trim_layer_order[:layer_trim_num_layer]


condense_layer_num = condense_num_expert
condense_layer_order = list(
    filter(lambda x: x not in trim_layer_idxs, condense_layer_order))
condense_layer_idxs = condense_layer_order[:condense_layer_num]
print("trim layer idxs :{}".format(trim_layer_idxs))
print("condense layer idxs: {}".format(condense_layer_idxs))

prune_layer_idx_to_expert_idx = {}
for prune_layer_idx in condense_layer_idxs:
    prune_expert_idx_list = layer_idx_to_expert_idxs[prune_layer_idx][:prune_num_expert]
    prune_layer_idx_to_expert_idx[prune_layer_idx] = prune_expert_idx_list
print(f"prune layer to expert: {prune_layer_idx_to_expert_idx}")

def classify_pruned_experts(name, prune_layer_idx_to_expert_idx):
    '''model.layers.0.block_sparse_moe.experts.7.w1'''
    try:
        layer_idx = int(name.split(".")[2])
        expert_idx = int(name.split(".")[-2])
        if layer_idx in prune_layer_idx_to_expert_idx and expert_idx not in prune_layer_idx_to_expert_idx[layer_idx]:
            return True
    except:
        return False
    return False

def classify_layer_trim_experts(name, trim_layer_idxs):
    '''model.layers.0.block_sparse_moe.experts.7.w1'''
    try:
        layer_idx = int(name.split(".")[2])
        expert_idx = int(name.split(".")[-2])
        if layer_idx in trim_layer_idxs and "experts" in name:
            return True
    except:
        return False
    return False

for name, module in model.named_modules():
    if isinstance(module, (torch.nn.Linear)) and \
          classify_pruned_experts(name, prune_layer_idx_to_expert_idx):
        print("condense {}".format(name))
        for param in module.parameters():
            param.requires_grad = False
            param.data = torch.tensor(
                [[0.1]], dtype=param.dtype, device=param.device)
            
for name, module in model.named_modules():
    if isinstance(module, (torch.nn.Linear)) and \
          classify_layer_trim_experts(name, trim_layer_idxs):
        print("layer trim {}".format(name))
        for param in module.parameters():
            param.requires_grad = False
            param.data = torch.tensor(
                [[0.1]], dtype=param.dtype, device=param.device)

torch.cuda.empty_cache()
import time
print(
    f"Average memory used during inference: {torch.cuda.memory_allocated()/1024**2} MB")
print_trainable_parameters(model)

# Add padding token if not present
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    model.resize_token_embeddings(len(tokenizer))

# Preprocess the dataset


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=2000)


encoded_dataset = dataset.map(preprocess_function, batched=True)

# Function to get GPU memory usage


def get_gpu_memory_usage():
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
        encoding="utf-8"
    )
    gpu_memory = int(result.strip().split('\n')[0])
    return gpu_memory


def model_inference(model, inputs):
    with torch.no_grad():
        outputs = model(**inputs)


sample_batch = encoded_dataset.select(range(args.batch_size))
inputs = {key: torch.tensor(val).cuda() for key, val in sample_batch.to_dict(
).items() if key in tokenizer.model_input_names}
print(len(inputs))
for k, v in inputs.items():
    print(k, v.size())

memory_usages = []
inference_times = []
memory_usages_before = []
#
test_cuda = []
test_cuda_before = []
# Repeat the measurement
for _ in range(args.num_repeats):
    torch.cuda.synchronize()
    # before_memory_allocated = torch.cuda.memory_allocated()

    start_time = time.time()
    with torch.no_grad():
        output = model(**inputs)
    end_time = time.time()

    torch.cuda.synchronize()
    after_memory_allocated = torch.cuda.memory_allocated()
    memory_usages.append(after_memory_allocated)
    # memory_usages_before.append(before_memory_allocated)
    inference_time = end_time - start_time
    inference_times.append(inference_time)

    del output
    torch.cuda.empty_cache()  # Clear the cache to get more accurate measurements
    gc.collect()

# Calculate averages
average_memory_usage = np.mean(memory_usages)
# average_memory_usage_before=np.mean(memory_usages_before)
average_inference_time = np.mean(inference_times)
throughput = args.batch_size / average_inference_time

print(
    f"Average memory used during inference: {average_memory_usage/1024**2} MB")
print(f"Average inference time: {average_inference_time:.4f} seconds")
print(f"Throughput: {throughput:.2f} inferences/second")
