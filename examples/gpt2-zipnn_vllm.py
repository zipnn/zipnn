from zipnn import zipnn_safetensors;
from vllm import LLM
zipnn_safetensors();

llm = LLM("zipnn/gpt2-ZipNN")

prompt = "Once upon a time,"
outputs = llm.generate([prompt])
print(outputs[0].outputs[0].text)
