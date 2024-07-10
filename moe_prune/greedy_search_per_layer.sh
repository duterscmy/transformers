#!/bin/bash  
# 遍历0到63之间的每一个数字，代表要剪枝的层索引  
for layer_idx in $(seq 0 63); do  
    # 执行Python脚本，传递模型路径和要剪枝的层索引  
    python moe_prune/eval_ppl.py --model ./deepseek16b --prune-layer $layer_idx  
    # 可选：如果需要查看每个循环的输出，可以在命令后添加echo来打印当前层索引  
    echo "Evaluated prune layer: $layer_idx"  
done