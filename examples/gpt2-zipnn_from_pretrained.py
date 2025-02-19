from transformers import AutoModelForCausalLM, AutoTokenizer
from zipnn import zipnn_safetensors
zipnn_safetensors()

model = "zipnn/gpt2-ZipNN"
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForCausalLM.from_pretrained(model, variant="znn")

prompt = "Once upon a time,"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=10)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
