from multiprocessing import Manager

manager = Manager()
expert_idxs_list  = manager.list()
global_layer_list = manager.list()
prune_layer_list = manager.list()
layer_num_list = manager.list()
global_layer_list.append(0)
expert_idx_to_info = manager.dict()  # (layer_idx, expert_idx):[sum_weight, freq]