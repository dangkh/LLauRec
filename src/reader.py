import os
import requests
import json
import pandas as pd
import csv
from tqdm import tqdm
# from proj_config import TrainConfig
import gzip
from collections import defaultdict
import numpy as np
import pickle
from scipy.sparse import coo_matrix
from sentence_transformers import SentenceTransformer

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.dumps(eval(l))

class DataConfig:
    random_seed: int = 1009
    data: str = 'baby'# ["baby", "ml-1m", "book"]
    device: str = 'cpu' # 'cuda:0'

if __name__ == "__main__":
    cfg = DataConfig()
    # dataPath = ".\data" + str(cfg.data)
    dataPath = os.path.join("data", str(cfg.data))
    if not os.path.exists(dataPath):
        os.makedirs(dataPath)
    if cfg.data == 'baby':
        linkmeta = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Baby.json.gz"
        link5cores = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Baby_5.json.gz"
        pathmeta = os.path.join("data", "baby", "baby.json.gz")
        metaUnzip = os.path.join("data", "baby", "baby.json")
        path5cores = os.path.join("data", "baby", "review_baby.json.gz")
        coreUnzip = os.path.join("data", "baby", "review_baby.json")
        print(f"Downloading Baby metadata...{path5cores} -- {pathmeta}")
        if os.path.exists(path5cores):
            print(f"{path5cores} File already exists. Skipping download.")
        else:
            r = requests.get(link5cores, stream=True)
            if r.status_code == 200:
                with open(path5cores, 'wb') as f:
                    f.write(r.raw.read())
                print(f"Downloaded {path5cores} successfully.")
            else:
                print(f"Failed to download {path5cores}. Status code: {r.status_code}")
        if os.path.exists(pathmeta):
            print(f"{pathmeta} File already exists. Skipping download.")
        else:
            r = requests.get(linkmeta, stream=True)
            if r.status_code == 200:
                with open(pathmeta, 'wb') as f:
                    f.write(r.raw.read())
                print(f"Downloaded {pathmeta} successfully.")
            else:
                print(f"Failed to download {pathmeta}. Status code: {r.status_code}")
        

        if os.path.exists(coreUnzip):
            print(f"{coreUnzip} File already exists. Skipping extraction.")
        else:
            print("Processing Baby dataset...")
            with open(coreUnzip, 'w', encoding='utf-8') as f:
                for l in parse(path5cores):
                    f.write(l+'\n')
        
        if os.path.exists(metaUnzip):
            print(f"{metaUnzip} File already exists. Skipping extraction.")
        else:
            print("Processing Baby dataset...")
            with open(metaUnzip, 'w', encoding='utf-8') as f:
                for l in parse(pathmeta):
                    f.write(l+'\n')

        print("Reading dataset...")
        jsons = []
        for line in open(coreUnzip, 'r', encoding='utf-8'):
            jsons.append(json.loads(line))
        
        print("Example: ", jsons[0])
        items = set()
        users = set()
        for js in jsons:
            items.add(js['asin'])
            users.add(js['reviewerID'])
        print(f"Total users: {len(users)}, Total items: {len(items)}")
        num_users = len(users)
        num_items = len(items)

        itemid = {}
        with open(os.path.join(dataPath, 'item_list.txt'), 'w', encoding='utf-8') as f:
            for i, it in enumerate(items):
                itemid[it] = i
                f.write(it + '\t' + str(i) + '\n')
        
        userid = {}
        with open(os.path.join(dataPath, 'user_list.txt'), 'w', encoding='utf-8') as f:
            for i, us in enumerate(users):
                userid[us] = i
                f.write(us + '\t' + str(i) + '\n')

        ui = defaultdict(list)
        for j in jsons:
            u = userid[j['reviewerID']]
            it = itemid[j['asin']]
            ui[u].append(it)

        with open(os.path.join(dataPath, 'user_item.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(ui))
        # text feature extraction can be added here
        jsons_meta = []
        for line in open(metaUnzip, 'r', encoding='utf-8'):
            jsons_meta.append(json.loads(line))
        rawtexts = {}
        for jm in jsons_meta:
            if jm['asin'] in itemid:
                textst = ' '
                if 'categories' in jm:
                    for cats in jm['categories']:
                        for cat in cats:
                            textst += cat + ' '
                if 'title' in jm:
                    textst += jm['title'] + ' '
                if 'brand' in jm:
                    textst += jm['brand'] + ' '
                if 'description' in jm:
                    textst += jm['description'] + ' '
                rawtexts[itemid[jm['asin']]] = textst.replace('\n', ' ')
        texts = []
        for text in rawtexts.values():
            texts.append(text)
        bert_model = SentenceTransformer('all-MiniLM-L6-v2', device=cfg.device)
        textEmbd = bert_model.encode(texts, show_progress_bar=True, device=cfg.device)
        np.save(os.path.join(dataPath, 'item_text_embd.npy'), textEmbd)

    elif cfg.data == 'ml-1m':
        pass
    elif cfg.data == 'book':
        pass
    
    print("Data preparation completed.")
    # spliting data into train/test sets can be added here
    rows_train, cols_train, data_train = [], [], []
    rows_val, cols_val, data_val = [], [], []
    rows_test, cols_test, data_test = [], [], []
    
    for u, ui_list in ui.items():
        if len(ui_list) < 10:
            testval = np.random.choice(len(ui_list), 2, replace=False)
        else:
            testval = np.random.choice(len(ui_list), int(len(ui_list) * 0.2), replace=False)
        test_idx = testval[:len(testval)//2]
        val_idx = testval[len(testval)//2:]
        train_idx = [i for i in range(len(ui_list)) if i not in testval]
        for idx in train_idx:
            rows_train.append(u)
            cols_train.append(ui_list[idx])
            data_train.append(1)
        for idx in val_idx:
            rows_val.append(u)
            cols_val.append(ui_list[idx])
            data_val.append(1)
        for idx in test_idx:
            rows_test.append(u)
            cols_test.append(ui_list[idx])
            data_test.append(1)
    train_matrix = coo_matrix((data_train, (rows_train, cols_train)), shape=(len(users), len(items)))
    val_matrix = coo_matrix((data_val, (rows_val, cols_val)), shape=(len(users), len(items)))
    test_matrix = coo_matrix((data_test, (rows_test, cols_test)), shape=(len(users), len(items)))
    with open(os.path.join(dataPath, 'train_matrix.pkl'), 'wb') as f:
        pickle.dump(train_matrix, f)
    with open(os.path.join(dataPath, 'val_matrix.pkl'), 'wb') as f:
        pickle.dump(val_matrix, f)
    with open(os.path.join(dataPath, 'test_matrix.pkl'), 'wb') as f:
        pickle.dump(test_matrix, f)
    print("Train/Val/Test matrices saved.")