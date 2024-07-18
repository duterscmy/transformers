### install
``` 
git clone --branch greedy_search_expert https://github.com/duterscmy/transformers.git
pip install ./transformers
``` 

### download model and add files
``` 
pip install huggingface_hub openpyxl
python download_deepseek16b.py
cp deepseek_model/modeling_deepseek.py deepseek_model/dynamic_weight.json ./deepseek16b
``` 
### greedy search expert per layer
```
python moe_prune/eval_ppl.py --model ./deepseek16b --prune-layer 0  # 测试下是否有结果
sh moe_prune/greedy_search_per_layer.sh  # 耗时1-2day
zip -r greedy_search_expert_output.zip greedy_search_expert_output/
``` 
