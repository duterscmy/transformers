#!/bin/bash  
# 遍历0到63之间的每一个数字，代表要剪枝的层索引
set -ex
input=datasets/c4-train.00000-of-01024.100.json
output_path=./greedy_search_layer_mixtral

mkdir -p $output_path
for layer_idx in $(seq 0 31); do  
    # 执行Python脚本，传递模型路径和要剪枝的层索引  
    python moe_prune/greedy_search_expert_mixtral.py \
        --dynamic-weight-file mixtral/dynamic_weights.mixtral.json \
        --model ../autodl-tmp/mixtral/ai-modelscope/mixtral/ \
        --num-layer 32 \
        --num-expert 8 \
        --prune-layer $layer_idx \
        --load-in-8bit > $output_path/${layer_idx}.search.log
    # 可选：如果需要查看每个循环的输出，可以在命令后添加echo来打印当前层索引  
    echo "Evaluated prune layer: $layer_idx"  
done