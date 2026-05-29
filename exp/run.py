#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: On Oversquashing project authors
"""

import sys
import os
import random
import csv
import fcntl
from pathlib import Path

import torch
import torch.nn.functional as F
import git
import numpy as np
import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from parser import get_parser

from torch_geometric.loader import DataLoader

from utils.factory import build_model, build_dataset
from utils.utils import set_seed, reset_wandb_env


def append_results(results_file, row):
    if results_file is None:
        return

    path = Path(results_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        'task',
        'experiment_label',
        'topology',
        'dataset',
        'model',
        'sheaf_variant',
        'sheaf_normalize_output',
        'sheaf_add_self_loops',
        'stalk_dim',
        'hidden_dim',
        'mpnn_layers',
        'synthetic_size',
        'distance',
        'seed',
        'epochs',
        'best_acc',
        'final_acc',
        'best_loss',
        'final_loss',
        'best_epoch',
        'sha',
        'wandb_project',
        'wandb_group',
        'wandb_run_name',
    ]
    with path.open('a+', encoding='utf-8', newline='') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0, os.SEEK_END)
        header = handle.tell() == 0
        writer = csv.DictWriter(handle, fieldnames=columns)
        if header:
            writer.writeheader()
        writer.writerow({column: row.get(column, '') for column in columns})
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    logits = model(data.x, data.edge_index)
    out = logits[data.mask]
    loss = F.cross_entropy(out, data.y)
    loss.backward()
    optimizer.step()
    pred = out.detach().max(1)[1]
    acc = pred.eq(data.y).sum().item() / data.mask.sum().item()
    return acc, loss.detach().cpu().item()


def test(model, data):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        mask = data.mask
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y).sum().item() / mask.sum().item()
        loss = F.cross_entropy(logits[mask], data.y)
        return acc, loss.detach().cpu().item()

def run_exp(args, dataset, model):
    data = dataset
    model = model.to(args.device)

    assert len(data) == args.synth_train_size + args.synth_test_size
    # 5k for training
    # 500 for test
    # following Bodnar et al. CW networks
    train_data = data[:args.synth_train_size]
    test_data = data[args.synth_train_size:]

    train_loader = DataLoader(train_data,
                              shuffle=True,
                              batch_size=args.bs)


    test_loader = DataLoader(test_data,
                              shuffle=False,
                              batch_size=len(test_data))

    optimizer = torch.optim.Adam(model.parameters(),
                                 weight_decay=args.weight_decay,
                                 lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                    mode="max",
                    factor=0.5,
                    patience=10,
                    min_lr=1e-5)


    best_test_acc = 0.0
    best_test_loss = float('inf')
    best_epoch = 0
    bad_counter = 0
    keep_running = True
    for epoch in range(args.epochs):
        train_acc, train_loss, test_acc, test_loss = [], [], [], []

        #train
        for batch in train_loader:
            batch = batch.to(args.device)
            train_acc_batch, train_loss_batch = train(model, optimizer, batch)
            train_acc.append(train_acc_batch)
            train_loss.append(train_loss_batch)

        #test
        for batch in test_loader:
            test_acc_batch, test_losses_batch = test(model, batch.to(args.device))
            test_acc.append(test_acc_batch)
            test_loss.append(test_losses_batch)

        train_acc = np.mean(train_acc)
        train_loss = np.mean(train_loss)
        test_acc = np.mean(test_acc)
        test_loss = np.mean(test_loss)

        res_dict = {
            'train_acc': train_acc,
            'train_loss': train_loss,
            'test_acc': test_acc,
            'test_loss': test_loss,
        }

        wandb.log(res_dict, step=epoch)

        new_best_trigger = test_acc > best_test_acc if args.stop_strategy == 'acc' else test_loss < best_test_loss
        if new_best_trigger:
            best_test_acc = test_acc
            best_test_loss = test_loss
            best_epoch = epoch
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.early_stopping:
            keep_running = False if test_acc < args.min_acc else True
            break

        if test_acc == 1.0:
            print("Perfect accuracy, stopping to save resources.")
            break

        scheduler.step(test_acc)
        print(f"Epochs: {epoch} | Best epoch: {best_epoch}")
        print(f"Test acc: {test_acc:.4f}")
        print(f"Best test acc: {best_test_acc:.4f}")

    wandb.log({'best_test_acc': best_test_acc,
               'best_test_loss': best_test_loss,
               'best_epoch': best_epoch,
               'final_test_acc': test_acc,
               'final_test_loss': test_loss})
    keep_running = False if test_acc < args.min_acc else True

    return test_acc, best_test_acc, test_loss, best_test_loss, keep_running, best_epoch


def build_wandb_kwargs(args):
    entity = None if args.entity in {None, '', 'none', 'None'} else args.entity
    tags = [tag.strip() for tag in args.wandb_tags.split(',') if tag.strip()]
    topology = 'crossed-ring' if args.dataset == 'RING' and args.add_crosses else args.dataset.lower()
    tags.extend(['graph-transfer', topology, args.model])
    return {
        'project': args.wandb_project,
        'entity': entity,
        'group': args.wandb_group,
        'name': args.wandb_run_name,
        'mode': args.wandb_mode,
        'tags': tags,
    }



if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if args.torch_num_threads is not None:
        torch.set_num_threads(args.torch_num_threads)
    if args.torch_num_interop_threads is not None:
        torch.set_num_interop_threads(args.torch_num_interop_threads)

    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    reset_wandb_env()

    args.device = torch.device(
        f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    set_seed(args.seed)

    model_cls = build_model(args)
    dataset = build_dataset(args)
    random.shuffle(dataset)

    # Add extra arguments
    args.sha = sha
    args.hidden_channels = args.hidden_dim
    args.num_layers = args.mpnn_layers
    topology = 'crossed-ring' if args.dataset == 'RING' and args.add_crosses else args.dataset.lower()
    args.topology = args.topology_label or topology
    args.distance = args.synthetic_size // 2 + 1 if args.dataset == 'LOLLIPOP' else args.synthetic_size // 2

    results = []
    wandb.init(config=vars(args), **build_wandb_kwargs(args))
    print(args)

    test_acc, best_test_acc, test_loss, best_test_loss, keep_running, best_epoch = run_exp(
        args, dataset, model_cls)
    results.append([test_acc, test_loss, best_test_acc, best_test_loss])
    test_acc_mean, test_loss_mean, best_test_acc_mean, best_test_loss_mean = np.mean(results, axis=0)
    test_acc_mean *= 100.0
    best_test_acc_mean *= 100.0
    test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100.0


    wandb_results = {'final_loss': test_loss_mean,
                 'final_acc': test_acc_mean,
                 'final_acc_std': test_acc_std,
                 'best_acc': best_test_acc_mean,
                 'best_loss': best_test_loss_mean,
                 'best_epoch': best_epoch,
                 'keep_running': keep_running}
    wandb.log(wandb_results)
    append_results(args.results_file, {
        'task': 'graph-transfer',
        'experiment_label': args.experiment_label,
        'topology': args.topology,
        'dataset': args.dataset,
        'model': args.model,
        'sheaf_variant': args.sheaf_variant if args.model == 'nsd' else '',
        'sheaf_normalize_output': args.sheaf_normalize_output if args.model == 'nsd' else '',
        'sheaf_add_self_loops': args.sheaf_add_self_loops if args.model == 'nsd' else '',
        'stalk_dim': args.stalk_dim if args.model == 'nsd' else '',
        'hidden_dim': args.hidden_dim,
        'mpnn_layers': args.mpnn_layers,
        'synthetic_size': args.synthetic_size,
        'distance': args.distance,
        'seed': args.seed,
        'epochs': args.epochs,
        'best_acc': best_test_acc_mean,
        'final_acc': test_acc_mean,
        'best_loss': best_test_loss_mean,
        'final_loss': test_loss_mean,
        'best_epoch': best_epoch,
        'sha': sha,
        'wandb_project': args.wandb_project,
        'wandb_group': args.wandb_group,
        'wandb_run_name': args.wandb_run_name,
    })
    wandb.finish()

    model_name = args.model
    print(f'{model_name} on {args.dataset} | SHA: {sha}')
    print(
        f'Test acc: {test_acc_mean:.4f} +/- {test_acc_std:.4f} | Test loss: {test_loss_mean:.4f}')
