import transformers
import torch

from zipnn import zipnn_hf

def main():
    zipnn_hf()
    model = "royleibov/Llama-3.1-8B-ZipNN-Compressed"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    print("Model loaded")

    messages = [
        {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "user", "content": "Who are you?"},
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=256,
    )
    print(outputs[0]["generated_text"][-1])

if __name__ == "__main__":
    main()
