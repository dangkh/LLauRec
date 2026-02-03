import pandas as pd
import yaml
import argparse
import numpy as np
import os
from unsloth import FastLanguageModel
import json
from unsloth.chat_templates import train_on_responses_only

from trl import SFTTrainer, SFTConfig
# from transformers import TrainingArguments
# from unsloth import is_bfloat16_supported
from datasets import Dataset
from datasets import load_dataset
import torch
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # tránh num_proc=68
device_map = {"": local_rank}
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
print(f"[rank {local_rank}] cuda device = {torch.cuda.current_device()} / {torch.cuda.get_device_name(0)}")


def formatting_prompts_func(examples):
   convos = examples["conversations"]
   texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
   return { "text" : texts, }


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', '-d', type=str, default='book', help='name of datasets')
	parser.add_argument('--LLM', type=str, default='Llama', help='name of LLM to use: Llama or Gemma, Qwen')
	parser.add_argument('--prompt_profile', '-pp', type=bool, default=True, help='ablation: item profile in prompt or not')
	parser.add_argument('--prompt_candidate', '-pc', type=bool, default=True, help='use candidate prompt or not')
	args, _ = parser.parse_known_args()
	print(args)

	fourbit_models = [
		"unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit", # Qwen 14B 2x faster
		"unsloth/Qwen3-4B-Thinking-2507-unsloth-bnb-4bit",
		"unsloth/Qwen3-8B-unsloth-bnb-4bit",
		"unsloth/Qwen3-14B-unsloth-bnb-4bit",
		"unsloth/Qwen3-32B-unsloth-bnb-4bit",

		# 4bit dynamic quants for superior accuracy and low memory use
		"unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
		"unsloth/Phi-4",
		"unsloth/Llama-3.1-8B",
		"unsloth/Llama-3.2-3B",
		"unsloth/orpheus-3b-0.1-ft-unsloth-bnb-4bit" # [NEW] We support TTS models!
	] # More models at https://huggingface.co/unsloth

	model, tokenizer = FastLanguageModel.from_pretrained(
		model_name = "unsloth/Qwen3-4B-Instruct-2507",
		max_seq_length = 4096, # Choose any for long context!
		load_in_4bit = True,  # 4 bit quantization to reduce memory
		load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
		full_finetuning = False, # [NEW!] We have full finetuning now!
		device_map = device_map,
		# token = "hf_...", # use one if using gated models
	)

	model = FastLanguageModel.get_peft_model(
		model,
		r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
		target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
						"gate_proj", "up_proj", "down_proj",],
		lora_alpha = 16,
		lora_dropout = 0, # Supports any, but = 0 is optimized
		bias = "none",    # Supports any, but = "none" is optimized
		# [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
		use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
		random_state = 3407,
		use_rslora = False,  # We support rank stabilized LoRA
		loftq_config = None, # And LoftQ
	)
	with open("src/prompts.yaml", "r") as f:
		all_prompts = yaml.safe_load(f)

	tuningP, systemP1, systemP2 = "tuning", "sys", "user"
	if args.prompt_candidate:
		tuningP, systemP1, systemP2 = "tuning_candidate", "sys_candidate", "user"
	tun_prompt = all_prompts[args.dataset][tuningP]
	sys_prompt = all_prompts[args.dataset][systemP1]
	
	dataPath = f"./data/{args.dataset}/candidate_{args.prompt_candidate}_profile_{args.prompt_profile}_tuningData.jsonl"
	dataset = load_dataset("json", data_files=dataPath)
	datalist = []
	for sample in dataset['train']:
		tmp = {
			"conversations": [
				{"role": "user", "content": sys_prompt + sample["userprompt"]},
				{"role": "assistant", "content": sample["answer"]},
			]
		}
		datalist.append(tmp)
	dataset = Dataset.from_list(datalist)
	from unsloth.chat_templates import get_chat_template
	tokenizer = get_chat_template(
		tokenizer,
		chat_template = "qwen3-instruct",
	)
	from unsloth.chat_templates import standardize_data_formats
	dataset = standardize_data_formats(dataset)
	dataset = dataset.map(formatting_prompts_func, batched = True, num_proc=0)

	trainer = SFTTrainer(
		model = model,
		tokenizer = tokenizer,
		train_dataset = dataset,
		eval_dataset = None, # Can set up evaluation!
		dataset_num_proc = 0,
		args = SFTConfig(
			dataset_text_field = "text",
			per_device_train_batch_size = 1,
			gradient_accumulation_steps = 4, # Use GA to mimic batch size!
			warmup_steps = 5,
			num_train_epochs = 1, # Set this for 1 full training run.
			# max_steps = 300,
			learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
			logging_steps = 1,
			optim = "adamw_torch",
			weight_decay = 0.001,
			lr_scheduler_type = "linear",
			seed = 3407,
			report_to = "none", # Use TrackIO/WandB etc
			local_rank= local_rank,
		),
	)
	trainer = train_on_responses_only(
		trainer,
		instruction_part = "<|im_start|>user\n",
		response_part = "<|im_start|>assistant\n",
	)
	tokenizer.decode(trainer.train_dataset[100]["input_ids"])
	tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[100]["labels"]]).replace(tokenizer.pad_token, " ")				

	trainer_stats = trainer.train()
	model.save_pretrained(f"qwen4B_it_model_{args.dataset}_candidate_{args.prompt_candidate}_profile_{args.prompt_profile}")  # Local saving
	tokenizer.save_pretrained(f"qwen4B_it_model_{args.dataset}_candidate_{args.prompt_candidate}_profile_{args.prompt_profile}")