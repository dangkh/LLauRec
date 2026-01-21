import pandas as pd
import yaml
import argparse
import numpy as np
import os
# from unsloth import FastLanguageModel
import json

# from trl import SFTTrainer
# from transformers import TrainingArguments
# from unsloth import is_bfloat16_supported
# from datasets import Dataset
# from datasets import load_dataset
# import torch
# local_rank = int(os.environ.get("LOCAL_RANK", 0))
# torch.cuda.set_device(local_rank)

# os.environ["TOKENIZERS_PARALLELISM"] = "false"  # tránh num_proc=68
# device_map = {"": local_rank}


def generate_conversation(examples):
	problems  = examples["problem"]
	solutions = examples["generated_solution"]
	conversations = []
	for problem, solution in zip(problems, solutions):
		conversations.append([
			{"role" : "user",      "content" : problem},
			{"role" : "assistant", "content" : solution},
		])
	return { "conversations": conversations, }

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', '-d', type=str, default='book', help='name of datasets')
	parser.add_argument('--LLM', type=str, default='Llama', help='name of LLM to use: Llama or Gemma, Qwen')
	parser.add_argument("--shard", type=int, default=0)
	parser.add_argument("--num_shards", type=int, default=1)
	parser.add_argument("--out", type=str, default="out.jsonl")
	args, _ = parser.parse_known_args()


	# fourbit_models = [
	# 	"unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit", # Qwen 14B 2x faster
	# 	"unsloth/Qwen3-4B-Thinking-2507-unsloth-bnb-4bit",
	# 	"unsloth/Qwen3-8B-unsloth-bnb-4bit",
	# 	"unsloth/Qwen3-14B-unsloth-bnb-4bit",
	# 	"unsloth/Qwen3-32B-unsloth-bnb-4bit",

	# 	# 4bit dynamic quants for superior accuracy and low memory use
	# 	"unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
	# 	"unsloth/Phi-4",
	# 	"unsloth/Llama-3.1-8B",
	# 	"unsloth/Llama-3.2-3B",
	# 	"unsloth/orpheus-3b-0.1-ft-unsloth-bnb-4bit" # [NEW] We support TTS models!
	# ] # More models at https://huggingface.co/unsloth

	# model, tokenizer = FastLanguageModel.from_pretrained(
	# 	model_name = "unsloth/Qwen3-4B-Instruct-2507",
	# 	max_seq_length = 4096, # Choose any for long context!
	# 	load_in_4bit = True,  # 4 bit quantization to reduce memory
	# 	load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
	# 	full_finetuning = False, # [NEW!] We have full finetuning now!
	# 	device_map = "balanced",
	# 	# token = "hf_...", # use one if using gated models
	# )

	# model = FastLanguageModel.get_peft_model(
	# 	model,
	# 	r = 32, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
	# 	target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
	# 					"gate_proj", "up_proj", "down_proj",],
	# 	lora_alpha = 32,
	# 	lora_dropout = 0, # Supports any, but = 0 is optimized
	# 	bias = "none",    # Supports any, but = "none" is optimized
	# 	# [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
	# 	use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
	# 	random_state = 3407,
	# 	use_rslora = False,  # We support rank stabilized LoRA
	# 	loftq_config = None, # And LoftQ
	# )
	
	dataset = load_dataset("json", data_files="data.jsonl")
	
	


