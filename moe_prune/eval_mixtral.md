由于要安装开发的transformers，需要新建一个虚拟环境

# 1.下载mixtral模型
```
# cache_dir需要改为合适的路径
# 路径下需要有至少190G空间
python moe_prune/download_mixtral.py  
```
# 2.安装lm-eval
```
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

# 3.评估mixtral模型
```
git clone https://github.com/duterscmy/transformers.git
cd transformers
pip install -e .

git checkout -b eval_mixtral origin/eval_mixtral

# transformers_dir和model_dir需要改成正确的路径
nohup sh moe_prune/loop_eval_mixtral.sh &

# 打包结果，结果文件名如：eval.e1.l18.batch8.log
zip eval_mixtral.zip eval.e*log
```