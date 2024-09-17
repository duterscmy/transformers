#!/bin/bash  
# 遍历0到63之间的每一个数字，代表要剪枝的层索引
set -ex
input=$1
output_path=$2

mkdir -p $output_path
for layer_idx in $(seq 1 23); do  
    # 执行Python脚本，传递模型路径和要剪枝的层索引  
    python moe_prune/greedy_search_expert.qwen.py \
        --input datasets/c4-train.00000-of-01024.cali.json \
        --output greedy_search_expert_qw \
        --dynamic-weight-file dynamic_weight.qwen.json \
        --model ../autodl-tmp/qw16/qwen/Qwen1___5-MoE-A2___7B/ \
        --num-layer 24 --num-expert 60 \
        --prune-layer $layer_idx > $output_path/${layer_idx}.search.log
    # 可选：如果需要查看每个循环的输出，可以在命令后添加echo来打印当前层索引  
    echo "Evaluated prune layer: $layer_idx"  
done