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

Convert dataset preproced in RLM [paper](https://arxiv.org/abs/2310.15950) to our standard.
```sh
python src/convert.py -d book
```

```sh
python src/preprocess.py -d book
```


```sh
python src/preprocess.py -d book
```

Run
```sh
python src/main.py -d book
```