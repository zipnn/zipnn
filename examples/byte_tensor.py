import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import zipnn
import gc

# Load
model_name = "ibm-granite/granite-3.0-8b-instruct"
device = "cpu"  # Or "cuda" if you have a GPU
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map=device)
print("Model downloaded and loaded successfully!")

# Init
comp_avg_tensor_list = []
decomp_avg_tensor_list = []
comp_avg_byte_list = []
decomp_avg_byte_list = []
state_dict = model.state_dict()
ITERATIONS=20

# Iterate (only 1 layer is needed)
for layer_name, tensor in state_dict.items():
    layer_size_mb = tensor.numel() * tensor.element_size() / (1024 ** 2)
    if layer_size_mb == 100.0: # Specific layer
        for threads in range(1, 9):
            comp_avg_byte = 0
            decomp_avg_byte = 0
            comp_avg_tensor = 0
            decomp_avg_tensor = 0
            for iteration in range(ITERATIONS):
                print(f"Iteration: {iteration}, Threads: {threads}")
                
                # Torch
                zpn = zipnn.ZipNN(input_format='torch', threads=threads)
                
                start_time = time.time()
                compressed_tensor = zpn.compress(tensor)
                comp_time = time.time() - start_time

                start_time = time.time()
                decompressed_tensor = zpn.decompress(compressed_tensor)
                decomp_time = time.time() - start_time

                if torch.equal(tensor, decompressed_tensor):
                    comp_avg_tensor += comp_time * (1/ITERATIONS)
                    decomp_avg_tensor += decomp_time * (1/ITERATIONS)

                # Byte
                byte_tensor = tensor.numpy().tobytes()
                zpn = zipnn.ZipNN()
                
                start_time = time.time()
                comp_byte = zpn.compress(byte_tensor)
                comp_time = time.time() - start_time
                
                start_time = time.time()
                decomp_byte = zpn.decompress(comp_byte)
                decomp_time = time.time() - start_time

                comp_avg_byte += comp_time * (1/ITERATIONS)
                decomp_avg_byte += decomp_time * (1/ITERATIONS)

            comp_avg_tensor_list.append(comp_avg_tensor)
            decomp_avg_tensor_list.append(decomp_avg_tensor)
            comp_avg_byte_list.append(comp_avg_byte)
            decomp_avg_byte_list.append(decomp_avg_byte)
            gc.collect()

        break # Only 1 layer is needed

threads_range = range(1, 9)
plt.figure(figsize=(10, 6))

plt.plot(threads_range, comp_avg_tensor_list, label='Tensor Compression', marker='o', linestyle='-', color='blue')
plt.plot(threads_range, decomp_avg_tensor_list, label='Tensor Decompression', marker='o', linestyle='--', color='blue')

plt.plot(threads_range, comp_avg_byte_list, label='Byte Compression', marker='x', linestyle='-', color='green')
plt.plot(threads_range, decomp_avg_byte_list, label='Byte Decompression', marker='x', linestyle='--', color='green')

plt.xlabel('Threads')
plt.ylabel('Time (seconds)')
plt.title('Compression and Decompression Times for Tensor and Byte Data')
plt.legend()
plt.tight_layout()
plt.show()
