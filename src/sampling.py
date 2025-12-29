import os
import requests
from math import nan
import json
import pandas as pd
import csv
import ast

from tqdm import tqdm
import yaml
import gzip


import argparse
from tqdm import tqdm

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', '-d', type=str, default='Books', help='name of datasets')
	# parser.add_argument('--out', type=str, default='./data/meta_filtered.csv', help='output csv path')
	args, _ = parser.parse_known_args()

	# =========================
	# Load meta data
	# =========================
	meta_data = []
	file_path = f'./data/meta_{args.dataset}_filtered.csv'
	metaDF = pd.read_csv(file_path)
	unique_meta_asin = set(metaDF['asin'])
	print(f"[Meta] Unique ASINs: {len(unique_meta_asin)}")

	# =========================
	# Load review data
	# =========================
	file_path = f'./data/reviews_{args.dataset}_5_filtered.csv'
	review5DF = pd.read_csv(file_path)
	# review5DF = pd.DataFrame(review_data)
	review_asin_set = set(review5DF['asin'])
	unique_users = review5DF['reviewerID'].unique()

	print(f"[Review] Unique ASINs: {len(review_asin_set), len(unique_users)}")

	# =========================
	# Sample 25% users
	# =========================
	sample_ratio = 0.01
	random_state = 42

	unique_users = review5DF['reviewerID'].unique()
	print(f"[Review] Total users: {len(unique_users)}")

	sampled_users = pd.Series(unique_users).sample(
		frac=sample_ratio,
		random_state=random_state
	).values

	print(f"[Review] Sampled users (25%): {len(sampled_users)}")

	review5DF_sampled = review5DF[
		review5DF['reviewerID'].isin(sampled_users)
	].reset_index(drop=True)

	# =========================
	# Save filtered review data
	# =========================
	out_review_path = f"./data/reviews_{args.dataset}_5_filtered_1k.csv"

	review5DF_sampled.to_csv(out_review_path, index=False)
	print(f"Saved filtered review file to: {out_review_path}")


