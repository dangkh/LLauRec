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
from sentence_transformers import SentenceTransformer
bertmodel = SentenceTransformer('all-MiniLM-L6-v2')


def overlap_items(list1, list2):
	return len(set(list1) & set(list2))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', '-d', type=str, default='book', help='name of datasets')
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

	# encode text_feat to embeddings and save as .npy
	
	text_embeddings = bertmodel.encode(merged_df['text_feat'].tolist(), show_progress_bar=True)
	text_embeddings = np.array(text_embeddings)
	text_feat_npy_path = os.path.join(dir, f'text_feat_original.npy')
	np.save(text_feat_npy_path, text_embeddings)

	# save prf embeddings as .npy in the order of itemID
	
	prf_text_embeddings = bertmodel.encode(prf_text, show_progress_bar=True)
	prf_text_embeddings = np.array(prf_text_embeddings)
	prf_text_feat_npy_path = os.path.join(dir, f'profile_text_feat.npy')
	np.save(prf_text_feat_npy_path, prf_text_embeddings)


