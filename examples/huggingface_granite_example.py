from transformers import AutoModelForCausalLM, AutoTokenizer

from zipnn import zipnn_hf

def main():
    zipnn_hf()
    # Load the tokenizer
    model = "royleibov/granite-7b-instruct-ZipNN-Compressed"
    tokenizer = AutoTokenizer.from_pretrained(model)

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model)

    # Example text
    input_text = "Hello, how are you?"

    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt")

    # Generate text
    outputs = model.generate(**inputs)

    # Decode the generated text
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(decoded_output)

if __name__ == "__main__":
    main()
