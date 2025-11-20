import os
import argparse
from utils.quick_start import quick_start

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='VLIF', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='book', help='name of datasets')

    config_dict = {
        'dropout': [0.2],
        'reg_weight': [0.001],
        'learning_rate': [0.003],
        'n_layers': [2],
        'gpu_id': 0,
    }

    args, _ = parser.parse_known_args()

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)

