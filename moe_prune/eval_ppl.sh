#! /bin/bash

# test
# python eval_ppl.py --score-mode test_route

# prune by l1, one layer
python eval_ppl.py --score-mode l1 --output-dir output_l1 --prune-one-layer

# prune by weightwatcher alpha, one layer
python eval_ppl.py --score-mode ww_alpha --output-dir output_ww_alpha --prune-one-layer

# prune by random, one layer
python eval_ppl.py --score-mode random --output-dir output_random --prune-one-layer

# prune by l1
python eval_ppl.py --score-mode l1 --output-dir output_l1 

# prune by weightwatcher alpha
python eval_ppl.py --score-mode ww_alpha --output-dir output_ww_alpha

# prune by random
python eval_ppl.py --score-mode random --output-dir output_random

