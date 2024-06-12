#! /bin/bash

# test
# python eval_ppl.py --score-mode test_route

# prune by l1, one layer
python moe_prune/eval_ppl.py --score-mode l1 --layer-mode one_layer

# prune by weightwatcher alpha, one layer
python moe_prune/eval_ppl.py --score-mode ww_alpha --layer-mode one_layer

# prune by random, one layer
python moe_prune/eval_ppl.py --score-mode random --layer-mode one_layer

# prune by l1
python moe_prune/eval_ppl.py --score-mode l1 --layer-mode jump_layers

# prune by weightwatcher alpha
python moe_prune/eval_ppl.py --score-mode ww_alpha --layer-mode jump_layers

# prune by random
python moe_prune/eval_ppl.py --score-mode random --layer-mode jump_layers

