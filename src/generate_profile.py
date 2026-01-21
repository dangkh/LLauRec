from math import nan
import pandas as pd
from tqdm import tqdm
import yaml
import argparse
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
import random
from unsloth import FastLanguageModel
import torch
import json
from helper import build_item_item_knn, get_itemDesc, getUser_Interaction

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


def generate_summary(model, tokenizer, system_prompt, content):
	messages = [
		{"role": "system", "content": system_prompt},
		{"role" : "user", 
		"content" : content}
	]
	input_text = tokenizer.apply_chat_template(
		messages,
		tokenize = False,
		add_generation_prompt = True, # Must add for generation
	)
	inputs = tokenizer(
		input_text,
		return_tensors = "pt",
		add_special_tokens = True,
	).to("cuda")

	output = model.generate(
		**inputs,
		max_new_tokens = 500, # Increase for longer outputs!
		temperature = 0.7, top_p = 0.8, top_k = 20, # For non thinking
		do_sample = False
	)
	generated_tokens = output[0][inputs["input_ids"].shape[-1]:]
	return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', '-d', type=str, default='book', help='name of datasets')
	parser.add_argument('--LLM', type=str, default='Llama', help='name of LLM to use: Llama or Gemma, Qwen')
	parser.add_argument("--shard", type=int, default=0)
	parser.add_argument("--num_shards", type=int, default=1)
	parser.add_argument("--out", type=str, default="out.jsonl")
	args, _ = parser.parse_known_args()

	# =========================
	# Load meta data
	# =========================
	meta_data = []
	file_path = f'./data/{args.dataset}/fullMeta_{args.dataset}.csv'
	metaDF = pd.read_csv(file_path)
	metaDF = pd.DataFrame(metaDF)
	unique_meta_asin = set(metaDF['asin'])
	print(f"[Meta] Unique ASINs: {len(unique_meta_asin)}")

	file_path = f'./data/{args.dataset}/{args.dataset}.inter'
	interDF = pd.read_csv(file_path, sep="\t", usecols=['userID', 'itemID', 'x_label'])
	interDF['userID'] = interDF['userID'].astype(int)
	interDF['itemID'] = interDF['itemID'].astype(int)

	# =========================
	# Preparing for users
	# =========================
	# 1. get user interactions in interDF in x_label==0

	user_interactions = getUser_Interaction(interDF)

	# # # 2. for each user, get closest user by item overlap
	# # user_top1_similar = {}
	# # for uid in tqdm(user_interactions.keys()):
	# # 	u_items = user_interactions[uid]
	# # 	max_overlap = -1
	# # 	top1_uid = -1
	# # 	for other_uid in user_interactions.keys():
	# # 		if other_uid == uid:
	# # 			continue
	# # 		other_u_items = user_interactions[other_uid]
	# # 		overlap = overlap_items(u_items, other_u_items)
	# # 		if overlap > max_overlap:
	# # 			max_overlap = overlap
	# # 			top1_uid = other_uid
	# # 	user_top1_similar[uid] = (top1_uid, max_overlap)


	# =========================
	# Profiling for user
	# =========================
	with open("src/prompts.yaml", "r") as f:
		all_prompts = yaml.safe_load(f)
	sys_prompt = all_prompts[args.dataset]['user']


	itemDesc = get_itemDesc(metaDF)

	# for each item, find top-k similar items
	top_k = 10
	item_item_path = f'./data/{args.dataset}/item_top{top_k}item.npy'
	if os.path.exists(item_item_path):
		print(f"{item_item_path} exists, skip building item-item knn.")
		item_kitem = np.load(item_item_path)
	else:
		raise ValueError(f"{item_item_path} does not exist, please run preprocess.py to build it.")


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
		device_map = "balanced",
		# token = "hf_...", # use one if using gated models
	)

	model = FastLanguageModel.get_peft_model(
		model,
		r = 32, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
		target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
						"gate_proj", "up_proj", "down_proj",],
		lora_alpha = 32,
		lora_dropout = 0, # Supports any, but = 0 is optimized
		bias = "none",    # Supports any, but = "none" is optimized
		# [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
		use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
		random_state = 3407,
		use_rslora = False,  # We support rank stabilized LoRA
		loftq_config = None, # And LoftQ
	)


	user_profiles = {}
	checkarray = []
	listUser = list(user_interactions.keys())
	users = listUser[args.shard::args.num_shards]

	user_profile_path = f'./data/{args.dataset}/usr_prf_{args.LLM}_{args.shard}.json'
	if os.path.exists(user_profile_path):
		with open(user_profile_path, 'r', encoding='utf-8') as f:
			user_profiles = json.load(f)
		print(f"Loaded existing user profiles from {user_profile_path}, current size: {len(user_profiles)}")
	for uid in tqdm(users):
		if str(uid) in user_profiles:
			continue
		u_items = user_interactions[uid]
		itemInfo = ""
		for item in u_items[-10:]:
			itemInfo += itemDesc[item]
		
		summary = generate_summary(model, tokenizer, sys_prompt, itemInfo)
		user_profiles[str(uid)] = { "summary": summary }

		if (len(user_profiles) + 1) % 50 == 0:
			with open(user_profile_path, 'w', encoding='utf-8') as f:
				json.dump(user_profiles, f, ensure_ascii=False, indent=4)

	with open(user_profile_path, 'w', encoding='utf-8') as f:
		json.dump(user_profiles, f, ensure_ascii=False, indent=4)
	
	# stat for candidate
	# print(np.mean(checkarray), np.min(checkarray), np.max(checkarray))
	


