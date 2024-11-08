from transformers import RobertaTokenizer, RobertaModel, pipeline
from zipnn import zipnn_hf

def main():
    zipnn_hf()
    # Load the model
    model_id = "royleibov/roberta-base-ZipNN-Compressed"

    unmasker = pipeline('fill-mask', model=model_id)

    print(unmasker("Hello I'm a <mask> model."))
    # tokenizer = RobertaTokenizer.from_pretrained(model_id)
    # model = RobertaModel.from_pretrained(model_id)

    # text = "Replace me by any text you'd like."
    # encoded_input = tokenizer(text, return_tensors='pt')
    # output = model(**encoded_input)

    # print(output)

if __name__ == "__main__":
    main()
