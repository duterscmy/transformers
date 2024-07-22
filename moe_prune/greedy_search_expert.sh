#!/bin/bash  
# 遍历0到63之间的每一个数字，代表要剪枝的层索引  
for layer_idx in $(seq 0 5); do  
    # 执行Python脚本，传递模型路径和要剪枝的层索引  
    python moe_prune/greedy_search_expert.py --model /root/autodl-tmp/deepseek-ai/deepseek-moe-16b-base --prune-layer $layer_idx  > greedy_search_expert_output_by_jl/${layer_idx}.search.log
    # 可选：如果需要查看每个循环的输出，可以在命令后添加echo来打印当前层索引  
    echo "Evaluated prune layer: $layer_idx"  
done