export HF_DATASETS_OFFLINE=1
set -ex
hyper_file="../autodl-tmp/deepseek-ai/deepseek-moe-16b-base/exp_hyper.py"
for prune_expert_num in 6 0; do
    for prune_layer_num in 6 9 12 15; do
        eval_log=logs/finetune.commonsense.e${prune_expert_num}.l${prune_layer_num}.log
        output_dir=../autodl-tmp/deepseek-ai/finetune.commonsense.e${prune_expert_num}.l${prune_layer_num}
        hyper="prune_layer_num=${prune_layer_num};num_route_experts=${prune_expert_num}"
        echo $hyper
        echo $hyper > $hyper_file

        python moe_prune/finetune_weight_all.py \
        --input datasets/common_reasoning.train.json \
        --model ../autodl-tmp/deepseek-ai/deepseek-moe-16b-base/ \
        --output-dir $output_dir \
        --max-length 128 \
        --lr 5e-5 \
        --score-mode greedy_jl \
        --prune-num-expert $prune_expert_num \
        --prune-num-layer $prune_layer_num \
        --batch-size 16 > $eval_log
    done
done