set -ex
transformers_dir=/root/transformers
hyper_file="${transformers_dir}/src/transformers/models/mixtral/exp_hyper.py"

for prune_expert_num in 1 2; do
  for prune_layer_num in 3 6 9 12 15 18 21; do
    eval_log=eval.e${prune_expert_num}.l${prune_layer_num}.batch8.log
    hyper="prune_layer_num=${prune_layer_num};trim_layer_num=0;num_route_experts=${prune_expert_num}"
    echo $hyper
    echo $hyper > $hyper_file

    lm_eval --model hf \
    --model_args pretrained=/root/autodl-tmp/mixtral/ai-modelscope/mixtral,load_in_8bit=True,trust_remote_code=True \
    --device cuda \
    --tasks hellaswag,piqa \
    --batch_size 8 \
    --output_path output.json \
    --trust_remote_code > $eval_log

    done
done