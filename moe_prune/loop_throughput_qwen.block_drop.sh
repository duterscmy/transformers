
hyper_file="/root/transformers/src/transformers/models/qwen2_moe/exp_hyper.py"

for prune_layer_num in 3 6 9 12 15; do
    eval_log=qwen_model/logs_throughput/block_drop.l${prune_layer_num}.log
    hyper="prune_layer_num=0;num_route_experts=0;trim_layer_num=${prune_layer_num}"
    echo $hyper
    echo $hyper > $hyper_file

    python moe_prune/evaluateThrougthputAndMemory_qwen_block.py \
        --model_name ../autodl-tmp/qwen_model/qwen/Qwen1___5-MoE-A2___7B/ \
        --layer-trim-num-layer ${prune_layer_num} > $eval_log
done