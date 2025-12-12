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
	parser.add_argument('--out', type=str, default='./data/meta_filtered.csv', help='output csv path')
	args, _ = parser.parse_known_args()

	# =========================
	# Load meta data
	# =========================
	meta_data = []
	with gzip.open(f'./data/meta_{args.dataset}.json.gz', 'rt') as f:
		for line in f:
			meta_data.append(ast.literal_eval(line))

	metaDF = pd.DataFrame(meta_data)
	metaDF_filtered = metaDF[
		["asin", "title", "brand", "description", "imUrl", "categories"]
	].copy()

	unique_meta_asin = set(metaDF_filtered['asin'])
	print(f"[Meta] Unique ASINs: {len(unique_meta_asin)}")

	# =========================
	# Load review data
	# =========================
	review_data = []
	with gzip.open(f"./data/reviews_{args.dataset}_5.json.gz", "rt") as f:
		for line in f:
			review_data.append(json.loads(line))

	review5DF = pd.DataFrame(review_data)
	review_asin_set = set(review5DF['asin'])

	print(f"[Review] Unique ASINs: {len(review_asin_set)}")

	# =========================
	# Filter meta by review ASINs
	# =========================
	metaDF_kept = metaDF_filtered[
		metaDF_filtered['asin'].isin(review_asin_set)
	].reset_index(drop=True)

	print(f"[Filtered Meta] Remaining ASINs: {metaDF_kept['asin'].nunique()}")

	# =========================
	# Save to CSV
	# =========================
	metaDF_kept.to_csv(args.out, index=False)
	print(f"Saved filtered meta data to: {args.out}")

	print("\n[Review DF] Head:")
	print(review5DF.head())

	print("\n[Review DF] Columns:")
	print(review5DF.columns)

	review_asin_set = set(review5DF['asin'])
	print(f"[Review] Unique ASINs: {len(review_asin_set)}")


	# =========================
	# Remove low-score reviews
	# =========================
	min_score = 3

	print(f"[Review] Before score filter: {len(review5DF)}")

	review5DF = review5DF[
		review5DF['overall'] >= min_score
	].reset_index(drop=True)

	print(f"[Review] After score filter (>= {min_score}): {len(review5DF)}")


	# =========================
	# Sample 25% users
	# =========================
	sample_ratio = 0.25
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

	print(f"[Review] Reviews after sampling: {len(review5DF_sampled)}")
	print(review5DF_sampled.head())

