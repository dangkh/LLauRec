import os
import requests
from math import nan
import json
import pandas as pd
import csv
import ast

from tqdm import tqdm
import yaml


import argparse
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', '-d', type=str, default='Books', help='name of datasets')
	args, _ = parser.parse_known_args()	

	data = []
	with gzip.open(f'./meta_{args.dataset}.json.gz', 'rt') as f:
		for line in f:
			data.append(ast.literal_eval(line))

	metaDF = pd.DataFrame(data)
	metaDF_filtered = metaDF[["asin", "title", "brand", "description", "imUrl", "categories"]].copy()

	unique_Meta_asin = metaDF_filtered['asin'].unique()
	print(f"Number of unique ASINs: {len(unique_Meta_asin)}")

	data = []
	with gzip.open(f"./review_{args.dataset}_5.json.gz", "r") as f:
	  for line in f:
		data.append(json.loads(line))
	
	review5DF = pd.DataFrame(data)
	print(review5DF.columns)

	unique_asin = review5DF['asin'].unique()
	print(f"Number of unique ASINs: {len(unique_asin)}")