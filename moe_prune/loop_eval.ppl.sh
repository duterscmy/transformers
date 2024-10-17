set -ex

hyper_file="/mnt/fast/nobackup/users/ly0008/caomingyu/deepseek-ai/deepseek-moe-16b-base/exp_hyper.py"
export HF_DATASETS_OFFLINE=1

prune_type="condense"
#0 26 7 12 19 20 22 23

for prune_expert_num in 0;do
  for prune_layer_num in $(seq 0 26); do
    hyper="prune_layer_idx=${prune_layer_num};num_route_experts=${prune_expert_num};prune_type=\"${prune_type}\""
    echo $hyper
    echo $hyper > $hyper_file

    eval_log=ppl/eval.per_layer.${prune_type}.e${prune_expert_num}.l${prune_layer_num}.log
    python moe_prune/evaluate_ppl.py --input /root/transformers/datasets/c4-train.00000-of-01024.100.json \
        --model /root/autodl-tmp/deepseek-ai/deepseek-moe-16b-base > $eval_log
    

    done
done