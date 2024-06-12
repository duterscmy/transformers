#! /bin/bash

# test
# python eval_ppl.py --score-mode test_route

# prune by l1, one layer
python eval_ppl.py --score-mode l1 --layer-mode one_layer

# prune by weightwatcher alpha, one layer
python eval_ppl.py --score-mode ww_alpha --layer-mode one_layer

# prune by random, one layer
python eval_ppl.py --score-mode random --layer-mode one_layer

# prune by l1
python eval_ppl.py --score-mode l1 --layer-mode jump_layers

# prune by weightwatcher alpha
python eval_ppl.py --score-mode ww_alpha --layer-mode jump_layers

# prune by random
python eval_ppl.py --score-mode random --layer-mode jump_layers

