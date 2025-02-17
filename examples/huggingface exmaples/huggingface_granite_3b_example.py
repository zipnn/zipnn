from transformers import AutoModelForCausalLM, AutoTokenizer

from zipnn import zipnn_hf

def main():
    zipnn_hf()
    # Load the tokenizer
    model = "royleibov/granite-3b-code-base-128k-ZipNN-Compressed"

    device = "cuda" # or "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model)

    model = AutoModelForCausalLM.from_pretrained(
        model, 
        device_map=device, 
        torch_dtype="auto", 
        trust_remote_code=True,
    )

    print("Model loaded")

    model.eval()
    # change input text as desired
    input_text = "def generate():"
    # tokenize the text
    input_tokens = tokenizer(input_text, return_tensors="pt")
    # transfer tokenized inputs to the device
    for i in input_tokens:
        input_tokens[i] = input_tokens[i].to(device)
    # generate output tokens
    output = model.generate(**input_tokens)
    # decode output tokens into text
    output = tokenizer.batch_decode(output)
    # loop over the batch to print, in this example the batch size is 1
    for i in output:
        print(i)

if __name__ == "__main__":
    main()
