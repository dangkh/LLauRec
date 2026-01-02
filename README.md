# LLAUREC
pytorch implementation for ""

```
VIRAL/
│
├── data/
│   └── <dataset_name>/
│       ├── train/test/valid_mat.pkl # train/valid/test set (sparse matrix)
│       ├── <dataset_name>_5.json
│       ├── i_id_mapping.csv
│       └── meta_<dataset_name>.json
├── README.md
├── src
│	├──get_text_feat.py              # Script to generate text embeddings
│	├──README.md
│   └─requirements.txt
```

<!-- run convert.py to convert data from RLMRec to our format -->
If not exist .inter, run convert.py
