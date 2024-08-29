from utils import print_trainable_parameters, \
    classify_shared_experts, \
    classify_remained_experts, \
    classify_pruned_experts
import gc
import argparse
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
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

parser.add_argument("--score-mode", type=str, default="greedy_jl", help="层间对专家排序的指标")
parser.add_argument("--prune-num-expert", default=6, type=int,
                    help="剪枝后剩余的expert数量")
parser.add_argument("--prune-num-layer", default=9, type=int,
                    help="剪枝后剩余的layer数量")
args = parser.parse_args()

score_mode = args.score_mode
prune_num_expert = args.prune_num_expert
prune_num_layer = args.prune_num_layer

# Load a sample of the Wiki dataset
dataset = load_dataset("json",
                       data_files="datasets/c4-train.00000-of-01024.head2k.json",
                       trust_remote_code=True,
                       split='train')
print(dataset)

# Load the model and tokenizer
# ignore_mismatched_sizes=True
model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        load_in_8bit = True
    )

tokenizer = AutoTokenizer.from_pretrained(
    args.model_name, trust_remote_code=True)

for param in model.parameters():
    param.requires_grad = False

print(
    f"Average memory used during inference: {torch.cuda.memory_allocated()/1024**2} MB")
print_trainable_parameters(model)

model.model.layers = model.model.layers[:-prune_num_layer]

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
