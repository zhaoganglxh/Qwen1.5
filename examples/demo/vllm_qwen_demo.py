from torch import float16

from vllm_wrapper import vLLMWrapper

#model = vLLMWrapper('/media/zhaogang/4T2-2(大语言模型)/HuggingFace/models/qwen/Qwen1___5-7B-Chat-GPTQ-Int8', tensor_parallel_size=1,gpu_memory_utilization=0.9,dtype="float16",quantization="gptq")
# model = vLLMWrapper('Qwen/Qwen-7B-Chat-Int4', tensor_parallel_size=1, dtype="float16")
model = vLLMWrapper('/media/zhaogang/4T2-2(大语言模型)/HuggingFace/models/qwen/Qwen1___5-7B-Chat', tensor_parallel_size=1, dtype="float16")
response, history = model.chat(query="你好", history=None)
print(response)
response, history = model.chat(query="给我讲一个年轻人奋斗创业最终取得成功的故事。", history=history)
print(response)
response, history = model.chat(query="给这个故事起一个标题", history=history)
print(response)