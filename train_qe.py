
import argparse
from modeling import QualityEstimator, QualityEstimatorForTrain
from transformers import AutoTokenizer, AutoConfig
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from dataset import generate_dataset
from tqdm import tqdm
from collections import OrderedDict
import json
from accelerate import Accelerator
import numpy as np
import random

def train(
    model: QualityEstimatorForTrain,
    loader: DataLoader,
    optimizer,
    epoch: int,
    accelerator: Accelerator
) -> float:
    model.train()
    log = {
        'loss': 0
    }
    with tqdm(enumerate(loader), total=len(loader), disable=not accelerator.is_main_process) as pbar:
        for _, batch in pbar:
            with accelerator.accumulate(model):
                loss = model(**batch)
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                log['loss'] += loss.item()
            
                if accelerator.is_main_process:
                    pbar.set_description(f'[Epoch {epoch}] [TRAIN]')
                    pbar.set_postfix(OrderedDict(
                        loss=loss.item()
                    ))
    return {k: v/len(loader) for k, v in log.items()}

def valid(model: QualityEstimatorForTrain,
    loader: DataLoader,
    epoch: int,
    accelerator: Accelerator
) -> float:
    model.eval()
    log = {
        'loss': 0
    }
    with torch.no_grad():
        with tqdm(enumerate(loader), total=len(loader), disable=not accelerator.is_main_process) as pbar:
            for _, batch in pbar:
                with accelerator.accumulate(model):
                    loss = model(**batch)
                    # loss = outputs['loss']

                    log['loss'] += loss.item()
                    
                    if accelerator.is_main_process:
                        pbar.set_description(f'[Epoch {epoch}] [VALID]')
                        pbar.set_postfix(OrderedDict(
                            loss=loss.item()
                        ))
    return {k: v/len(loader) for k, v in log.items()}

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.restore_dir is not None:
        config = json.load(open(os.path.join(args.restore_dir, 'impara_config.json')))
        model_id = config['model_id']
        qe_model = QualityEstimator(model_id)
        qe_model.load_state_dict(torch.load(os.path.join(args.restore_dir, 'pytorch_model.bin')))
        current_epoch = config['epoch'] + 1
        min_valid_loss = config['min_valid_loss']
        log_dict = json.load(open(os.path.join(args.restore_dir, '../log.json')))
    else:
        model_id = args.model_id
        config = AutoConfig.from_pretrained(model_id)
        qe_model = QualityEstimator(model_id)
        current_epoch = 0
        min_valid_loss = 1e9
        log_dict = {}
    model = QualityEstimatorForTrain(qe_model)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_dataset = generate_dataset(
        file_path=args.train_file,
        tokenizer=tokenizer
    )
    valid_dataset = generate_dataset(
        file_path=args.valid_file,
        tokenizer=tokenizer
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    os.makedirs(os.path.join(args.outdir, 'best'), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, 'last'), exist_ok=True)
    accelerator = Accelerator(gradient_accumulation_steps=args.accumulation)
    model, optimizer, train_loader, valid_loader = accelerator.prepare(
        model, optimizer, train_loader, valid_loader
    )
    for epoch in range(current_epoch, args.epochs):
        train_log = train(model, train_loader, optimizer, epoch, accelerator)
        valid_log = valid(model, valid_loader, epoch, accelerator)
        
        if accelerator.is_main_process:
            log_dict[f'Epoch {epoch}'] = {
                'train_loss': train_log,
                'valid_loss': valid_log
            }
            if min_valid_loss > valid_log['loss']:
                torch.save(qe_model.state_dict(), os.path.join(args.outdir, 'best/pytorch_model.bin'))
                min_valid_loss = valid_log['loss']
                config_dict = {
                    'model_id': model_id,
                    'epoch': epoch,
                    'min_valid_loss': min_valid_loss
                }
                with open(os.path.join(args.outdir, 'best/impara_config.json'), 'w') as fp:
                    json.dump(config_dict, fp, indent=4)
    if accelerator.is_main_process:
        torch.save(qe_model.state_dict(), os.path.join(args.outdir, 'last/pytorch_model.bin'))
        config_dict = {
            'model_id': model_id,
            'epoch': epoch,
            'min_valid_loss': min_valid_loss
        }
        with open(os.path.join(args.outdir, 'last/impara_config.json'), 'w') as fp:
            json.dump(config_dict, fp, indent=4)
        with open(os.path.join(args.outdir, 'log.json'), 'w') as fp:
            json.dump(log_dict, fp, indent=4)
        print('Finish')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', default='bert-base-cased')
    parser.add_argument('--outdir', default='models/sample/')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--accumulation', type=int, default=1)
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--train_file', required=True)
    parser.add_argument('--valid_file', required=True)
    parser.add_argument('--restore_dir')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    main(args)
