set -ex

hyper_file="/mnt/fast/nobackup/users/ly0008/caomingyu/deepseek-ai/deepseek-moe-16b-base/exp_hyper.py"
export HF_DATASETS_OFFLINE=1
# num_route_experts = 0
# prune_layer_idx = 0
# prune_type = "layer_trim"
prune_type="condense"
#0 26 7 12 19 20 22 23
for prune_expert_num in 6;do
  for prune_layer_num in $(seq 0 26); do
    hyper="prune_layer_idx=${prune_layer_num};num_route_experts=${prune_expert_num};prune_type=\"${prune_type}\""
    echo $hyper
    echo $hyper > $hyper_file

    # 提交任务并等待结束
    job_id=$(condor_submit eval_boolq.submit | grep -oE '[0-9]+\.[0-9]+')
    echo "Submitted first job with ID: $job_id"

    # 使用 condor_wait 等待第一个任务完成
    condor_wait c${job_id}.log $job_id

    echo "First job completed."
    

    done
done