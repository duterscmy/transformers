set -ex

hyper_file="/root/autodl-tmp/deepseek-ai/deepseek-moe-16b-base/exp_hyper.py"

# num_route_experts = 0
# prune_layer_idx = 0
# prune_type = "layer_trim"
prune_type="condense"

for prune_expert_num in 0 6;do
  for prune_layer_num in 0 26 7 12 19 20 22 23; do
    eval_log=eval.per_layer.${prune_type}.e${prune_expert_num}.l${prune_layer_num}.batch8.log
    hyper="prune_layer_idx=${prune_layer_num};num_route_experts=${prune_expert_num};prune_type=\"${prune_type}\""
    echo $hyper
    echo $hyper > $hyper_file

    lm_eval --model hf \
    --model_args pretrained=/root/autodl-tmp/deepseek-ai/deepseek-moe-16b-base/,trust_remote_code=True \
    --device cuda \
    --tasks arc_challenge,boolq,hellaswag,mmlu,openbookqa,piqa,rte,winogrande \
    --batch_size 32 \
    --output_path output.json \
    --trust_remote_code > $eval_log

    done
done