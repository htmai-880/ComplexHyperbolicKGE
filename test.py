# Inherited from https://github.com/HazyResearch/KGEmb
"""Evaluation script."""

import argparse
import json
import os
import torch

import models 
from datasets.kg_dataset import KGDataset
from utils.train import avg_both, format_metrics

parser = argparse.ArgumentParser(description="Test")
parser.add_argument(
    '--model_dir',
    help="Model path" 
)
KGHOME='./'
LOG_DIR=os.path.join(KGHOME, 'logs')
# DATA_PATH = '/KGEdata'
DATA_PATH=os.path.join(KGHOME, 'data')

def test(model_dir):
    # load config
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config = json.load(f)
    args = argparse.Namespace(**config)
    
    # create dataset
    dataset_path = os.path.join(DATA_PATH, args.dataset)
    dataset = KGDataset(dataset_path, False)
    test_examples = dataset.get_examples("test")
    filters = dataset.get_filters()

    # load pretrained model weights
    if "GCN" in args.model or "GAT" in args.model:
        model = getattr(models, args.model)(args, dataset)
    else:
        model = getattr(models, args.model)(args)
    # device = 'cuda'
    # model.to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pt'), map_location='cpu'))
    model.eval()
    
    # eval
    test_metrics = avg_both(*model.compute_metrics(test_examples, filters, batch_size=500))
    return test_metrics

if __name__ == "__main__":
    args = parser.parse_args()
    test_metrics = test(args.model_dir)
    print(format_metrics(test_metrics, split='test'))

