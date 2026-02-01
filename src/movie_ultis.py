import ast
import os
import json
import pandas as pd


dataset = 'movie'
data_dir = f'./data/{dataset}/'
# load file item_prompt.json, user_prompt.json
item_prompt = {}
with open(os.path.join(data_dir, 'item_prompt.json'), 'r') as f:
    for _line in f.readlines():
        _data = json.loads(_line)
        item_prompt[_data["item_id"]] = _data["prompt"]
        
user_prompt = {}
with open(os.path.join(data_dir, 'user_prompt.json'), 'r') as f:
    for _line in f.readlines():
        _data = json.loads(_line)
        user_prompt[_data["user_id"]] = _data["prompt"]

# get number of users and items
num_users = len(user_prompt)
num_items = len(item_prompt)
print(f'Number of users: {num_users}, Number of items: {num_items}')

# metaDF = items['title', 'description']
def get_item_prompt(item_id):
    return item_prompt.get(item_id, "")

metaDF = []
for item_id in range(num_items):
    tmp = get_item_prompt(item_id)
    metaDF.append(ast.literal_eval(tmp))

# convert to DataFrame

metaDF = pd.DataFrame.from_dict(metaDF)
# print columns
print(metaDF.columns.to_list())
print(metaDF.head())
metaDF['profile'] = metaDF['description']
metaDF['text_feat'] = metaDF["title"] + ' ' + metaDF["description"]
save_path = os.path.join(data_dir, 'item_meta.csv')
metaDF.to_csv(save_path)
