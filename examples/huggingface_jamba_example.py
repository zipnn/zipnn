from transformers import AutoTokenizer, AutoModelForCausalLM

from zipnn import zipnn_hf

def main():
    zipnn_hf()
    # Load the tokenizer
    model = "royleibov/Jamba-v0.1-ZipNN-Compressed"
    tokenizer = AutoTokenizer.from_pretrained(model)

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model)
    model.eval()

    # change input text as desired
    prompt = "Write a code to find the maximum value in a list of numbers."

    # tokenize the text
    input_tokens = tokenizer(prompt, return_tensors="pt")
    # generate output tokens
    output = model.generate(**input_tokens, max_new_tokens=100)
    # decode output tokens into text
    output = tokenizer.batch_decode(output)
    # loop over the batch to print, in this example the batch size is 1
    for i in output:
        print(i)

if __name__ == "__main__":
    main()
