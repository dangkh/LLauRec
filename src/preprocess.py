import os
import random
import requests
from math import nan
import json
import pandas as pd
import csv
import ast
from tqdm import tqdm
import argparse
import pickle
import numpy as np
import yaml
import gzip
from datasets import Dataset
from helper import build_item_item_knn, get_itemDesc, get_profile_embeddings, getUser_Interaction

def overlap_items(list1, list2):
	return len(set(list1) & set(list2))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', '-d', type=str, default='book', help='name of datasets')
	parser.add_argument("--item_profile", type=bool, default=False, help='whether to use item profile or not')
	args, _ = parser.parse_known_args()

	dir = f'./data/{args.dataset}/'
	# =========================
	# Load meta data
	# =========================
	meta_csv_path = os.path.join(dir, f'meta_{args.dataset}.json.gz')
	iid_asin_path = os.path.join(dir, f"{args.dataset}_asin.json")
	iid_asin = {}
# -------- read JSON Lines --------
	records = []
	with open(iid_asin_path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if line:
				records.append(json.loads(line))

	iid_df = pd.DataFrame(records)   # columns: iid, asin
	iid_asin_set = set(iid_df['asin'].tolist())
	# print(iid_df.head())
	print(f"Number of items in iid_asin: {len(iid_asin_set)}")
	print(f"Sample iid_asin: {iid_df.sample(5)}")


	# =========================
	# Load train data
	# =========================
	meta_data = []
	file_path = f'./data/{args.dataset}/{args.dataset}.inter'
	interDF = pd.read_csv(file_path, sep="\t", usecols=['userID', 'itemID', 'x_label'])
	interDF['userID'] = interDF['userID'].astype(int)
	interDF['itemID'] = interDF['itemID'].astype(int)
	# metaDF = pd.DataFrame(interDF)
	num_users = interDF['userID'].nunique()
	print(num_users)

	users_by_label = {
    l: set(interDF[interDF['x_label'] == l]['userID'].unique())
    for l in [0, 1, 2]
	}

	print("0 ∩ 1:", len(users_by_label[0] & users_by_label[1]))
	print("1 ∩ 2:", len(users_by_label[1] & users_by_label[2]))
	print("0 ∩ 2:", len(users_by_label[0] & users_by_label[2]))


	# =========================
	# Load item data
	# =========================
	file_path = f'./data/{args.dataset}/itm_prf.pkl'
	with open(file_path, 'rb') as f:
		prf = pickle.load(f)
	
	# check all items in metaDF appear in prf
	meta_items = set(interDF['itemID'].unique())
	prf_items = set(prf.keys())
	print("Number of items in metaDF and prf:", len(meta_items & prf_items))
	print("Number of items only in metaDF:", len(meta_items - prf_items))

	prf_text = []
	for idx in prf_items:
		item_profile = prf[idx]['profile']
		prf_text.append(item_profile)

	# random a single sample of item profiles
	randomID = random.choice(list(prf_items))
	print("An item profile contains:", prf[randomID].keys(), "sample item:", prf[randomID])

	metaDF_filtered_path = os.path.join(dir, f'metaDF_filtered_{args.dataset}.csv')
	if os.path.exists(metaDF_filtered_path) is False:
		data = []
		with gzip.open(meta_csv_path, 'rt') as f:
			for line in tqdm(f):
				tmp = ast.literal_eval(line)
				if tmp['asin'] not in iid_asin_set:
					continue
				data.append(tmp)

		metaDF = pd.DataFrame(data)
		metaDF_filtered = metaDF[["asin", "title", "description"]].copy()
		# saving metaDF_filtered
		metaDF_filtered.to_csv(metaDF_filtered_path, index=False)
	else:
		metaDF_filtered = pd.read_csv(metaDF_filtered_path)
	# interDF: id, asin
	# metaDF_filtered: asin, title, description
	# merge interDF with metaDF_filtered
	# 

	print("interDF columns:", iid_df.columns.tolist())
	print(iid_df.head())

	print("\nmetaDF_filtered columns:", metaDF_filtered.columns.tolist())
	print(metaDF_filtered.head())


	merged_df = iid_df.merge(
		metaDF_filtered[["asin", "title", "description"]],
		on="asin",
		how="left"
	)
	merged_df = merged_df.sort_values(by='iid').reset_index(drop=True)
	
	
	# print number of missing both titles and descriptions
	num_missing_both = merged_df['title'].isnull() & merged_df['description'].isnull()
	print(f"Number of missing both titles and descriptions: {num_missing_both.sum()}")
	# print number of rows in merged_df
	print(f"Number of rows in merged_df: {merged_df.shape[0]}")

	# fill null titles or descriptions with empty string
	merged_df['title'] = merged_df['title'].fillna('')
	merged_df['description'] = merged_df['description'].fillna('')
	merged_df['profile'] = prf_text
	fullMeta_filtered_path = os.path.join(dir, f'fullMeta_{args.dataset}.csv')
	merged_df.to_csv(fullMeta_filtered_path, index=False)

	# create new column with combine title and description
	merged_df['text_feat'] = merged_df['title'] + ' ' + merged_df['profile']

	if args.item_profile:
		# save prf embeddings as .npy in the order of itemID
		text_embeddings = get_profile_embeddings(merged_df['text_feat'].tolist(), path = os.path.join(dir, f'text_feat_profile.npy'))
	else:	
		# encode text_feat to embeddings and save as .npy
		text_embeddings = get_profile_embeddings(merged_df['text_feat'].tolist(), path = os.path.join(dir, f'text_feat_original.npy'))
	
	top_k = 10
	item_kitem = build_item_item_knn(text_embeddings, top_k=top_k)
	item_item_path = f'./data/{args.dataset}/item_top{top_k}item.npy'
	np.save(item_item_path, item_kitem)

	user_interactions = getUser_Interaction(interDF)
	itemDesc = get_itemDesc(merged_df, merge=False)
	checkarray = []
	listUser = list(user_interactions.keys())

	with open("src/prompts.yaml", "r") as f:
		all_prompts = yaml.safe_load(f)
	tun_prompt = all_prompts['tuning']
	sys_prompt = all_prompts[args.dataset]['sys']

	tuningLLM_name = 'QwenTuning'
	dataset = []
	for uid in tqdm(listUser):
		u_items = user_interactions[uid]
		selected = u_items[-10:] 
		ground_truth = selected[-1]
		interacted = selected[:-1]
		itemInfo = ""
		for item in interacted:
			title, description = itemDesc[item]
			tmp = f"Title: {title}\nDescription: {description}\n\n"
			itemInfo += tmp

		candidates = item_kitem[ground_truth]
		listC = []
		for c in candidates:
			if c in u_items:
				continue
			listC.append(c)
		random.shuffle(listC)
		listC = listC[:3]
		checkarray.append(len(listC))
		candidateInfo = ""
		for c in listC:
			title, description = itemDesc[c]
			tmp = f"Title: {title}\nDescription: {description}\n\n"
			candidateInfo += tmp

		userprompt = tun_prompt.format(itemInfo, candidateInfo)
		answer = f"User may like: {itemDesc[ground_truth][1]}"
		dataset.append({
			"userprompt": userprompt,
			"systemprompt": sys_prompt,
			"answer": answer
		})

	
	dataset = Dataset.from_list(dataset)
	dataset.to_json(f"./data/{args.dataset}/tuningData.jsonl")
	# stat for candidate
	print(np.mean(checkarray), np.min(checkarray), np.max(checkarray))	