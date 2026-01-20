from math import nan
import pandas as pd
from tqdm import tqdm
import yaml
import argparse
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
import random
# from unsloth import FastLanguageModel
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

def build_item_item_knn(itemDesc, top_k=50, metric="cosine", return_scores=False):
    """
    itemDesc: np.ndarray shape (num_items, dim) hoặc list-of-embeddings
    top_k: số hàng xóm gần nhất (không tính chính nó)
    metric: "cosine" (khuyến nghị) hoặc "euclidean"
    return_scores: nếu True trả thêm độ tương đồng/ khoảng cách
    """
    X = np.asarray(itemDesc, dtype=np.float32)
    n = X.shape[0]
    if top_k >= n:
        raise ValueError(f"top_k must be < num_items (top_k={top_k}, num_items={n})")

    # Với cosine: sklearn trả về "cosine distance" = 1 - cosine_similarity
    nn = NearestNeighbors(n_neighbors=top_k + 1, metric=metric, algorithm="auto")
    nn.fit(X)

    _, indices = nn.kneighbors(X, return_distance=True)  # (n, top_k+1)

    # loại bỏ chính nó (thường ở vị trí 0)
    item_item = indices[:, 1:top_k+1].astype(np.int32)

    if not return_scores:
        return item_item

def getDescribe(vlmModel, tokenizer, link = None, title = None, sysPrompt = None, myPrompt = None):
    
	messages=[
		{"role": "system", "content": sysPrompt},
		{"role": "user", "content": myPrompt},
	]
	input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
	inputs = tokenizer(image, input_text, add_special_tokens = False, return_tensors = "pt").to(model.device)
	output = vlmModel.generate(**inputs, max_new_tokens=256, temperature=0.1, do_sample=False,)
	generated_tokens = output[0][inputs["input_ids"].shape[-1]:]
	output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
	return output_text

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', '-d', type=str, default='book', help='name of datasets')
	parser.add_argument('--LLM', type=str, default='Llama', help='name of LLM to use: Llama or Gemma, Qwen')
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

	user_interactions = {}
	for idx, row in tqdm(interDF.iterrows(), total=interDF.shape[0]):
		uid = row['userID']
		iid = row['itemID']
		label = row['x_label']
		if label != 0:
			continue
		if uid not in user_interactions:
			user_interactions[uid] = []
		user_interactions[uid].append(int(iid))

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
	print(sys_prompt)


	itemDesc = []
	for idx, row in tqdm(metaDF.iterrows(), total=metaDF.shape[0]):
		iid = row['iid']
		title = row['title']
		description = row['profile']
		itemDesc.append(f"Title: {title}\nDescription: {description}\n\n")

	# for each item, find top-k similar items
	top_k = 10
	item_item_path = f'./data/{args.dataset}/item_top{top_k}item.npy'
	if os.path.exists(item_item_path):
		print(f"{item_item_path} exists, skip building item-item knn.")
		item_kitem = np.load(item_item_path)
	else:
		embed_path = f'./data/{args.dataset}/text_feat_original.npy'
		item_embeddings = np.load(embed_path)
		item_kitem = build_item_item_knn(item_embeddings, top_k=top_k)
		np.save(item_item_path, item_kitem)


	user_profiles = {}
	checkarray = []
	for uid in tqdm(user_interactions.keys()):
		u_items = user_interactions[uid]
		itemInfo = ""
		for item in u_items:
			itemInfo += itemDesc[item]
		# candidates = item_kitem[u_items[-1]]
		# listC = []
		# for c in candidates:
		# 	if c not in u_items:
		# 		listC.append(c)
		# random.shuffle(listC)
		# listC = listC[:3]
		# checkarray.append(len(listC))
		# candidateInfo = ""
		# for c in listC:
		# 	candidateInfo += itemDesc[c]
		print(itemInfo)
		stop
	
	
	# stat for candidate
	print(np.mean(checkarray), np.min(checkarray), np.max(checkarray))



