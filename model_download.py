from modelscope.hub.snapshot_download import snapshot_download

model_dir = snapshot_download('qwen/Qwen1.5-7B-Chat-GPTQ-Int8',revision='master',cache_dir='/media/zhaogang/4T2-2(大语言模型)/HuggingFace/models')
print(model_dir)