import os, sys, json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time
import gc
import psutil
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

print(f"Python executable: {sys.executable}")
print(f"Pytorch version: {torch.__version__}")
print(f"Transformers version: {__import__('transformers').__version__}")
print(f"Benchmark timestamp: {datetime.now().isoformat()}")
print(f"=" * 60)

# model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
custom_cache_dir = "/dev/shm"
#test_prompt = "Provide one paragraph on best things about Japan"
test_prompt = "Explain the benefits of using Kubernetes for machine learning workloads:"

def get_system_info():
    # Get comprehensive system resources metadata
    info = {
            "cpu_cores": psutil.cpu_count(),
            "total_ram_gb": psutil.virtual_memory().total / (1024**3),
            "available_ram_gb": psutil.virtual_memory().available / (1024**3),
            "timestamp": datetime.now().isoformat()
            }
    
    print(f"CUDA device present - {torch.cuda.is_available()}")
    if torch.cuda.is_available() == True:
        info["cuda_version"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()
        info["gpus"] = []

        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            gpu_info = {
                    "id": i,
                    "name": gpu_props.name,
                    "memory_gb": gpu_props.total_memory / (1024**3),
                    "compute_capability": f"{gpu_props.major}.{gpu_props.minor}",
                    "multiprocessor_count": gpu_props.multi_processor_count
                    }
            info["gpus"].append(gpu_info)
            print(f"GPU {i}: {gpu_props.name}")
            print(f"Memory: {gpu_props.total_memory / (1024**3):.2f} GB")
            print(f"Compute capability: {gpu_props.major}.{gpu_props.minor}")
    else:
        print("XX - CUDA not available - check GPU drivers and pytorch installation")
        return None

    print(f"CPU cores: {info['cpu_cores']}")
    print(f"Total RAM: {info['total_ram_gb']:.2f} GB")
    print(f"Available RAM: {info['available_ram_gb']} GB")
    print(f"=" * 60)
    return info

def generate_text(prompt, max_length=200, temperature=0.7, top_p=0.8):
    """ Basic text generation test"""

    # Encode inputs
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)

    # Track GPU memory before generation
    memory_before = torch.cuda.memory_allocated() / 1e9
    start_time = time.time()

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
                )

    # Calculate metrics
    generation_time = time.time() - start_time
    memory_after = torch.cuda.memory_allocated() / 1e9

    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    new_text = response[len(prompt):].strip()

    print(f"GPU Name: {torch.cuda.get_device_properties(0).name}")
    print(f"Model Name: {model_name}")
    print(f"Generation time: {generation_time:.2f}s")
    print(f"Memory delta: {memory_after - memory_before:.3f} GB")
    print(f"Generated tokens: {len(outputs[0]) - len(inputs[0])}")
    print(f"Tokens/sec: {(len(outputs[0]) - len(inputs[0])) / generation_time:.1f}")
    
    return new_text


system_info = get_system_info()
if system_info is None:
    print("\n GPU setup required before continuing. Check environment configuration")
    exit()

print(f"Loading model - {model_name} ....")

# Loading Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=custom_cache_dir)

#Add padding token if none
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Quantization Config
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False)

#Load model
model_init_time = time.time()

#model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True, low_cpu_mem_usage=True, cache_dir=custom_cache_dir)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, device_map="auto", trust_remote_code=True, low_cpu_mem_usage=True, cache_dir=custom_cache_dir)

model_load_time = time.time() - model_init_time
print(f"Model loaded successfully ...")
print(f"Device: {model.device}")
print(f"Model memory: {model.get_memory_footprint() / 1e9:.2f} GB")
print(f"Model load time: {model_load_time:.2f}s")
print("=" * 60)
print("=")
print("=")

#Check GPU memory usage after model loading

gpu_memory = torch.cuda.memory_allocated() / 1e9
gpu_reserved = torch.cuda.memory_reserved() / 1e9
print(f"GPU Memory - Allocated: {gpu_memory:.2f} GB, Reserved: {gpu_reserved:.2f} GB")

result = generate_text(test_prompt, max_length=2000)
print(f"\n    Generated text:\n{result}")

# cleanup

del model
del tokenizer
torch.cuda.empty_cache()
gc.collect()

