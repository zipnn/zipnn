from transformers import AutoModelForCausalLM, AutoTokenizer

from zipnn import zipnn_hf

def main():
    zipnn_hf()
    # Load the tokenizer
    model = "royleibov/solar-pro-preview-instruct-ZipNN-Compressed"
    tokenizer = AutoTokenizer.from_pretrained(model)

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model,
                                                 device_map="cuda",  
                                                 torch_dtype="auto",  
                                                 trust_remote_code=True,)
    model.eval()
    print("Model loaded")

    # Apply chat template
    messages = [
        {"role": "user", "content": "Please, introduce yourself."},
    ]
    prompt = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
    
    # Generate text
    outputs = model.generate(prompt, max_new_tokens=512)
    print(tokenizer.decode(outputs[0]))

if __name__ == "__main__":
    main()
