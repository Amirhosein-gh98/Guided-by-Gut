import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Settings
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # or another causal-LM
# model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
dtype = torch.float16
seq_len = 16000

# Load model/tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Generate long input of repeated tokens
dummy_token_id = tokenizer.encode("Hello", add_special_tokens=False)[0]
input_ids = torch.full((1, seq_len + 100), dummy_token_id, dtype=torch.long).cuda()  # make longer than needed

# ✅ Trim to exactly 16,000 tokens
input_ids = input_ids[:, :seq_len]

# Forward pass with cache
with torch.no_grad():
    outputs = model(input_ids=input_ids, use_cache=True)
    past_key_values = outputs.past_key_values

# Compute KV cache size
total_bytes = sum(
    tensor.numel() * tensor.element_size()
    for pair in past_key_values
    for tensor in pair
)

# Output in GB
print(f"KV cache size for {seq_len} tokens: {total_bytes / (1024**3):.2f} GB")

