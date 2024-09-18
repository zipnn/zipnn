from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from zipnn import zipnn_hf

def main():
    zipnn_hf()
    # Load the tokenizer
    model = "royleibov/Phi-3.5-mini-instruct-ZipNN-Compressed"
    tokenizer = AutoTokenizer.from_pretrained(model)

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model)
    print("Model loaded")

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
        {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
        {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
    ]

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }

    output = pipe(messages, **generation_args)
    print(output[0]['generated_text'])

if __name__ == "__main__":
    main()
