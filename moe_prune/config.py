

import json


def get_layer_idx_order(prune_num_expert, score_mode):
    '''get prune layer order'''
    # ppl order pruning single layer
    if prune_num_expert == 0:
        # layer_idx_list_ppl_order = [11, 18, 7, 8, 2, 23, 10, 22, 13, 16,
        #                             15, 20, 24, 19, 25, 4, 6, 5, 3, 9, 21, 27, 17, 12, 26, 14, 1]
        # layer_idx_list_ppl_order = [layer-1 for layer in layer_idx_list_ppl_order]
        layer_idx_list_ppl_order = [10, 17, 22, 1, 12,
                                    21, 6, 15, 7, 19, 24, 9]  # greedy ppl
        layer_idx_list_ppl_order = [19, 12, 7, 23, 10, 14,
                                    1, 24, 17, 15, 9, 21, 18, 6, 26]  # greedy by jl
    elif prune_num_expert == 6 and score_mode == "random":
        # layer_idx_list_ppl_order = [11, 18, 7, 23, 15, 8, 10, 2, 22, 20,
        #                             24, 16, 13, 6, 3, 19, 25, 4, 5, 9, 21, 27, 17, 12, 26, 14, 1]
        # layer_idx_list_ppl_order = [layer-1 for layer in layer_idx_list_ppl_order]
        layer_idx_list_ppl_order = [10, 17, 22, 1, 9,
                                    6, 21, 15, 12, 14, 19, 7]
    elif prune_num_expert == 6 and score_mode == "l1":
        layer_idx_list_ppl_order = [5, 18, 11, 22, 8, 13, 10, 7, 23, 16,
                                    2, 20, 4, 24, 15, 19, 9, 3, 25, 6, 17, 1, 21, 27, 14, 12, 26]
        layer_idx_list_ppl_order = [
            layer-1 for layer in layer_idx_list_ppl_order]
        layer_idx_list_ppl_order = [4, 17, 21, 10, 22,
                                    12, 7, 15, 9, 19, 6, 18]
    elif prune_num_expert == 6 and score_mode == "distribution":
        layer_idx_list_ppl_order = [15, 10, 7, 18, 8, 2, 22, 16, 23, 11,
                                    20, 24, 13, 6, 19, 25, 4, 3, 5, 1, 27, 9, 21, 17, 12, 26, 14]
        layer_idx_list_ppl_order = [
            layer-1 for layer in layer_idx_list_ppl_order]
        layer_idx_list_ppl_order = [14, 9, 1, 21, 22,
                                    6, 17, 10, 12, 7, 15, 24]
    elif prune_num_expert == 6 and score_mode == "greedy_jl":
        layer_idx_list_ppl_order = [19, 15, 22, 10,
                                    12, 6, 14, 21, 26, 7, 17, 1, 24, 23, 9]  # greedy jl

    return layer_idx_list_ppl_order


def get_layer_idx_to_expert_idx(score_mode):
    '''get prune expert order per layer'''
    if score_mode == "l1":
        layer_idx_to_expert_idxs = json.load(
            open("deepseek_model/layer_idx_to_expert_idx.json", 'r'))
        layer_idx_to_expert_idxs = {
            int(key): value for key, value in layer_idx_to_expert_idxs.items()}
    elif score_mode == "ww_alpha":
        layer_idx_to_expert_idxs = json.load(
            open("deepseek_model/layer_idx_to_expert_idx.alpha.json", 'r'))
        layer_idx_to_expert_idxs = {
            int(key): value for key, value in layer_idx_to_expert_idxs.items()}
    elif score_mode == "distribution":
        layer_idx_to_expert_idxs = json.load(
            open("deepseek_model/layer_idx_to_expert_idx.distribution.json", 'r'))
        layer_idx_to_expert_idxs = {
            int(key): value for key, value in layer_idx_to_expert_idxs.items()}
    elif score_mode == "random":
        layer_idx_to_expert_idxs = json.load(
            open("deepseek_model/layer_idx_to_expert_idx.random.json", 'r'))
        layer_idx_to_expert_idxs = {
            int(key): value for key, value in layer_idx_to_expert_idxs.items()}
    elif score_mode == "greedy_jl":
        layer_idx_to_expert_idxs = json.load(
            open("deepseek_model/layer_idx_to_expert_idx.greedy_jl.json", 'r'))
        layer_idx_to_expert_idxs = {
            int(key): value for key, value in layer_idx_to_expert_idxs.items()}
    return layer_idx_to_expert_idxs


# load dynamic weights
dynamic_weights = {}
dynamic_weight_tmp = json.load(open("deepseek_model/dynamic_weight.json"))
for key, value in dynamic_weight_tmp.items():
    key = key.split("-")
    layer_idx = int(key[0])
    expert_idx = int(key[1])
    w = value[-1]
    dynamic_weights[(layer_idx, expert_idx)] = w
