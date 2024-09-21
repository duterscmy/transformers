
hyper_file="/root/transformers/src/transformers/models/qwen2_moe/exp_hyper.py"
for prune_expert_num in 0 4;do
    for prune_layer_num in 0 3 6 9 12 15; do
        eval_log=qwen_model/logs_throughput/eval.e${prune_expert_num}.l${prune_layer_num}.log
        hyper="prune_layer_num=${prune_layer_num};num_route_experts=${prune_expert_num};trim_layer_num=15"
        echo $hyper
        echo $hyper > $hyper_file

        python moe_prune/evaluateThrougthputAndMemory_qwen.py \
          --model_name ../autodl-tmp/qwen_model/qwen/Qwen1___5-MoE-A2___7B/ \
          --prune-num-expert ${prune_expert_num} \
          --condense-num-layer ${prune_layer_num} \
          --layer-trim-num-layer 0 > $eval_log
    done
done