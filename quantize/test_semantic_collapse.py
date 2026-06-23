"""Test 32L AttnMix quantized model for semantic collapse."""
import sys, time, json
sys.path.insert(0, '/home/lxy/workspace/RiverOne-QC-4B-v1-AQLM-miniViT/AQLM')
sys.path.insert(0, '/home/lxy/workspace/RiverOne-QC-4B-v1-AQLM-miniViT')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/home/lxy/workspace/RiverOne-QC-4B-v1-AQLM-miniViT/AQLM/RiverOne-QC-4B-v2-AQLM-2x16-32L-Attn14MLP16"

TEST_PROMPTS = [
    "你好，请用中文简单介绍一下自己。",
    "What is the capital of France? Answer in one sentence.",
    "请解释什么是量子计算，用一两句话。",
    "1 + 1 = ?",
    "请写一首关于春天的五言绝句。",
]

print("=" * 60)
print("32L AttnMix Semantic Collapse Test")
print("=" * 60)

t0 = time.time()
print(f"\n[1/3] Loading tokenizer from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"  vocab size: {tokenizer.vocab_size}")

print(f"\n[2/3] Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
load_time = time.time() - t0
print(f"  Load time: {load_time:.1f}s")

# Count total params
total = sum(p.numel() for p in model.parameters())
print(f"  Total params: {total/1e9:.2f}B")

# Count quantized params (AQLM codes)
quant_params = 0
for name, param in model.named_parameters():
    if 'codes' in name:
        quant_params += param.numel()
print(f"  AQLM code params: {quant_params/1e6:.2f}M")

print(f"\n[3/3] Running inference tests...\n")
model.eval()

for i, prompt in enumerate(TEST_PROMPTS):
    print(f"--- Test {i+1}: {prompt[:60]}... ---")
    try:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"  Response: {response[:300]}")
        
        # Basic collapse checks
        collapsed = False
        if len(response.strip()) < 2:
            collapsed = True
            print("  ⚠️  COLLAPSE: Empty/short response")
        elif response.count(response[:5]) > 10 and len(response) > 50:
            collapsed = True
            print("  ⚠️  COLLAPSE: Repetitive output detected")
        else:
            print("  ✅ Looks coherent")
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
    print()

print("=" * 60)
print(f"Total time: {time.time() - t0:.1f}s")
print("=" * 60)
