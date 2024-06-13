# %%
# %%
import json
import weightwatcher as ww
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoTokenizer, T5Tokenizer, AutoConfig, AutoModelForCausalLM, LogitsProcessorList, LogitsProcessor
from typing import List, Optional
from huggingface_hub import snapshot_download
import time

pytorch_checkpoint_path = "qw27"

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

# %%
config = AutoConfig.from_pretrained(pytorch_checkpoint_path)
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

# %%

model = AutoModelForCausalLM.from_pretrained(
    # "Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4",
    # "Qwen/Qwen1.5-MoE-A2.7B",
    "qw27",
    device_map=device_map,
    torch_dtype=torch.bfloat16,
)

for layer in model.model.layers:
    layer.mlp.split()
tokenizer = AutoTokenizer.from_pretrained("qw27")

# %% [markdown]
# l1

# %%
# 计算专家的L1范数
# key_to_score = {}

# for layer_idx, layer in enumerate(model.model.layers):
#   print("layer {}".format(layer_idx))
#   for expert_idx, expert in enumerate(layer.mlp.experts):
#     # print("expert {}".format(expert_idx))
#     gate_w = expert.gate_proj
#     up_w = expert.up_proj
#     down_w = expert.down_proj
#     gate_l1 = float(torch.norm(gate_w.weight.to(torch.float32).squeeze(), p=1))
#     up_l1 = float(torch.norm(up_w.weight.to(torch.float32).squeeze(), p=1))
#     down_l1 = float(torch.norm(down_w.weight.to(torch.float32).squeeze(), p=1))
#     key_to_score[(layer_idx, expert_idx, "gate")] = gate_l1
#     key_to_score[(layer_idx, expert_idx, "up")] = up_l1
#     key_to_score[(layer_idx, expert_idx, "down")] = down_l1

#   share_expert = layer.mlp.shared_expert
#   share_gate = share_expert.gate_proj.weight.to(torch.float32) # 5632*2048
#   share_up = share_expert.up_proj.weight.to(torch.float32) # 5632*2048
#   share_down = share_expert.down_proj.weight.to(torch.float32) # 2048*5632
#   # print(share_gate.size(), share_up.size(), share_down.size())
#   share_gate_list = torch.split(share_gate, 1408, dim=0)
#   share_up_list = torch.split(share_up, 1408, dim=0)
#   share_down_list = torch.split(share_down, 1408, dim=1)
#   expert_idx = 60
#   for g, u, d in zip(share_gate_list, share_up_list, share_down_list):
#     # print("expert {}".format(expert_idx))
#     key_to_score[(layer_idx, expert_idx, "gate")] = float(torch.norm(g.squeeze(), p=1))
#     key_to_score[(layer_idx, expert_idx, "up")] = float(torch.norm(u.squeeze(), p=1))
#     key_to_score[(layer_idx, expert_idx, "down")] = float(torch.norm(d.squeeze(), p=1))
#     expert_idx += 1

# # %%
# key_to_score
# import pickle
# pickle.dump(key_to_score, open("qw2.7B.l1.fp32.pkl", 'wb'))

# %% [markdown]
# alpha

# %%

# 创建新的模型，包含 32 个全连接层


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc_layers = nn.ModuleList(
            [nn.Linear(2048, 1408, bias=False) for _ in range(64)])

    def forward(self, x):
        outputs = []
        for i, tensor in enumerate(x):
            tensor = tensor.view(tensor.size(0), -1)
            output = self.fc_layers[i](tensor)
            outputs.append(output)
        return torch.cat(outputs, dim=1)


def expert_weights_to_fake_model(weights):
    tmp_model = MyModel()

    # 使用拆分的张量来初始化模型的参数
    for i, tensor in enumerate(weights):
        with torch.no_grad():
            tmp_model.fc_layers[i].weight.copy_(tensor.t())
    return tmp_model


# %%
# ww alpha
layer_idx_to_alpha_list = {}
for layer_idx in range(24):
    print(layer_idx)

    expert_weights = []
    for expert_idx in range(len(model.model.layers[layer_idx].mlp.experts)):
        w = model.model.layers[layer_idx].mlp.experts[expert_idx].down_proj.weight.to(
            torch.float32)
        expert_weights.append(w)

#   share_expert = model.model.layers[layer_idx].mlp.shared_expert
#   # share_gate = share_expert.gate_proj.weight.to(torch.float32) # 5632*2048
#   # share_up = share_expert.up_proj.weight.to(torch.float32) # 5632*2048
#   share_down = share_expert.down_proj.weight.to(torch.float32) # 2048*5632
#   # print(share_gate.size(), share_up.size(), share_down.size())
#   # share_gate_list = torch.split(share_gate, 1408, dim=0)
#   # share_up_list = torch.split(share_up, 1408, dim=0)
#   share_down_list = torch.chunk(share_down, 4, dim=1)
#   expert_weights.extend(share_down_list)

    print("num expert weights {}".format(len(expert_weights)))
    b = time.time()
    fake_model = expert_weights_to_fake_model(expert_weights)

    watcher = ww.WeightWatcher(model=fake_model)
    details = watcher.analyze(mp_fit=True, randomize=True)
    details
    alpha_list = list(details["alpha"])
    alpha_list
    print("get alpha for one layer {}".format(time.time()-b))
    print("len alpha list {}".format(len(alpha_list)))
    layer_idx_to_alpha_list[layer_idx] = alpha_list

# %%
layer_idx_to_alpha_list
json.dump(layer_idx_to_alpha_list, open("layer_idx_to_alpha_list.json", 'w'))
