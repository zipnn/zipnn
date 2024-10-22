from transformers import AutoModelForCausalLM, AutoTokenizer

from zipnn import zipnn_hf

def main():
    zipnn_hf()
    # Load the tokenizer
    model = "royleibov/granite-3.0-8b-instruct-ZipNN-Compressed"

    device = "cuda" # or "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model)

    model = AutoModelForCausalLM.from_pretrained(
        model, 
        device_map=device, 
        torch_dtype="auto", 
    )

    print("Model loaded")

    model.eval()
    # change input text as desired
    chat = [
        { "role": "user", "content": "Please list one IBM Research laboratory located in the United States. You should only output its name and location." },
    ]
    chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    # tokenize the text
    input_tokens = tokenizer(chat, return_tensors="pt").to(device)
    # generate output tokens
    output = model.generate(**input_tokens, 
                            max_new_tokens=100)
    # decode output tokens into text
    output = tokenizer.batch_decode(output)
    # print output
    print(output)


if __name__ == "__main__":
    main()
