
hyper_file="../autodl-tmp/deepseek-ai/deepseek-moe-16b-base/hyper.py"
prune_expert_num=1
for prune_layer_num in 3 6 9 12 15; do
    eval_log=eval.e${prune_expert_num}.l${prune_layer_num}.log
    hyper="prune_layer_num=${prune_layer_num}"
    echo $hyper
    echo $hyper > $hyper_file

    lm_eval --model hf \
    --model_args pretrained=/root/autodl-tmp/deepseek-ai/deepseek-moe-16b-base/,dtype="bfloat16",trust_remote_code=True \
    --device cuda \
    --tasks arc_challenge,boolq,hellaswag,mmlu,openbookqa,piqa,rte,winogrande \
    --batch_size 6 \
    --output_path output.json \
    --trust_remote_code > $eval_log
    done