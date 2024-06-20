#! /bin/bash

# test
# python eval_ppl.py --score-mode test_route
model_path=/root/autodl-tmp/deepseek-ai/deepseek-moe-16b-base/
# model_path=/root/autodl-tmp/qw27/
# prune by l1, one layer
python moe_prune/eval_ppl.py --score-mode l1 --layer-mode one_layer --model ${model_path} --num-layer 27

# prune by weightwatcher alpha, one layer
# python moe_prune/eval_ppl.py --score-mode ww_alpha --layer-mode one_layer --model ${model_path}

# prune by random, one layer
# python moe_prune/eval_ppl.py --score-mode random --layer-mode one_layer --model ${model_path}

# prune by l1
# python moe_prune/eval_ppl.py --score-mode l1 --layer-mode jump_layers --model ${model_path}

# prune by weightwatcher alpha
# python moe_prune/eval_ppl.py --score-mode ww_alpha --layer-mode jump_layers --model ${model_path}

# prune by random
python moe_prune/eval_ppl.py --score-mode random --layer-mode jump_layers --model ${model_path} --num-layer 27

