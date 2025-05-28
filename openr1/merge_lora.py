#!/usr/bin/env python
from pathlib import Path
import argparse
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    p = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    p.add_argument("--base_model",  required=True, help="HF name or local path to the *base* checkpoint")
    p.add_argument("--lora_path",   required=True, help="Directory with adapter_model.bin")
    p.add_argument("--output_dir",  required=True, help="Where to save the merged model")
    args = p.parse_args()

    print("⏳ Loading base model …")
    model     = AutoModelForCausalLM.from_pretrained(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print("⏳ Loading LoRA adapter …")
    peft_model   = PeftModel.from_pretrained(model, args.lora_path)

    print("🔧 Merging weights …")
    merged_model = peft_model.merge_and_unload()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("✅  Saved merged checkpoint to", args.output_dir)

if __name__ == "__main__":
    main()
