import torch
import librosa
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from zipnn import zipnn_hf


def main():
    zipnn_hf()
    
    LANG_ID = "en"
    MODEL_ID = "royleibov/wav2vec2-large-xlsr-53-english-ZipNN-Compressed"
    SAMPLES = 10

    test_dataset = load_dataset("mozilla-foundation/common_voice_11_0", LANG_ID, split=f"test[:{SAMPLES}]")

    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

    # Preprocessing the datasets.
    # We need to read the audio files as arrays
    def speech_file_to_array_fn(batch):
        speech_array, sampling_rate = librosa.load(batch["path"], sr=16_000)
        batch["speech"] = speech_array
        batch["sentence"] = batch["sentence"].upper()
        return batch

    test_dataset = test_dataset.map(speech_file_to_array_fn)
    inputs = processor(test_dataset["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_sentences = processor.batch_decode(predicted_ids)

    for i, predicted_sentence in enumerate(predicted_sentences):
        print("-" * 100)
        print("Reference:", test_dataset[i]["sentence"])
        print("Prediction:", predicted_sentence)
    
if __name__ == "__main__":
    main()
