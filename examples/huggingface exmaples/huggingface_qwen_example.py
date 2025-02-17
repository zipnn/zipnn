from PIL import Image
import requests
import torch
from torchvision import io
from typing import Dict
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from zipnn import zipnn_hf

def main():
    zipnn_hf()

    # Load the model in half-precision on the available device(s)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "royleibov/Qwen2-VL-7B-Instruct-ZipNN-Compressed", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("royleibov/Qwen2-VL-7B-Instruct-ZipNN-Compressed")

    print("Model loaded")

    # Image
    url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    image = Image.open(requests.get(url, stream=True).raw)

    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]


    # Preprocess the inputs
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    # Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'

    inputs = processor(
        text=[text_prompt], images=[image], padding=True, return_tensors="pt"
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    output_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    print(output_text)

if __name__ == "__main__":
    main()
