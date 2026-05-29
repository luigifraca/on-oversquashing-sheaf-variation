import argparse
import os
import random
import sys
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp')

import git
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import wandb
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.factory import NetFactory
from utils.utils import reset_wandb_env, set_seed


DEFAULT_MODELS = 'gcn,gat,nsd'


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {'y', 'yes', 't', 'true', 'on', '1'}:
        return True
    if value in {'n', 'no', 'f', 'false', 'off', '0'}:
        return False
    raise ValueError(f'Unrecognised boolean value {value}')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='NCI1')
    parser.add_argument('--root', type=str, default='data/tu')
    parser.add_argument('--models', type=str, default=DEFAULT_MODELS)
    parser.add_argument('--layers', type=int, default=10)
    parser.add_argument('--hidden_dim', type=int, default=5)
    parser.add_argument('--max_graphs', type=int, default=None)
    parser.add_argument('--num_sources', type=int, default=10)
    parser.add_argument('--seed', type=int, default=808)
    parser.add_argument('--output_dir', type=str,
                        default='signal_propagation_results')

    parser.add_argument('--sheaf_variant', type=str,
                        choices=['diagonal',
                                 'general',
                                 'orthogonal',
                                 'general_attention',
                                 'orthogonal_attention',
                                 'low_rank'],
                        default='general')
    parser.add_argument('--stalk_dim', type=int, default=4)
    parser.add_argument('--sheaf_alpha', type=float, default=1.0)
    parser.add_argument('--sheaf_rank', type=int, default=1)
    parser.add_argument('--sheaf_orth_strategy', type=str,
                        choices=['cayley', 'fasth'], default='cayley')
    parser.add_argument('--sheaf_add_self_loops', type=str2bool, default=True)
    parser.add_argument('--sheaf_normalize_output', type=str2bool,
                        default=True)
    parser.add_argument('--sheaf_jknet', type=str2bool, default=False)

    parser.add_argument('--entity', type=str, default=None)
    parser.add_argument('--wandb_project', type=str,
                        default='on-oversquashing-signal-propagation')
    parser.add_argument('--wandb_group', type=str,
                        default='signal-propagation-classical-vs-nsd')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--wandb_mode', type=str,
                        choices=['online', 'offline', 'disabled'],
                        default='online')
    parser.add_argument('--wandb_tags', type=str, default='')
    return parser


def parse_models(models):
    return [model.strip() for model in models.split(',') if model.strip()]


def initialize_architecture(model_name, in_channels, args):
    if model_name == 'nsd':
        from sheaf_mpnn import NSDModel, NSDVariant

        return NSDModel(
            in_channels=in_channels,
            out_channels=args.hidden_dim,
            stalk_dim=args.stalk_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.layers,
            variant=NSDVariant[args.sheaf_variant.upper()],
            alpha=args.sheaf_alpha,
            add_self_loops=args.sheaf_add_self_loops,
            orth_strategy=args.sheaf_orth_strategy,
            rank=args.sheaf_rank,
            normalize_output=args.sheaf_normalize_output,
            jknet=args.sheaf_jknet,
        )
    return NetFactory(arch=model_name, num_layers=args.layers,
                      dim_h=args.hidden_dim)


def prepare_graph(dataset_item):
    x = dataset_item.x
    if x is None:
        x = torch.ones((dataset_item.num_nodes, 1), dtype=torch.float32)
    return Data(
        x=x.float().clone(),
        edge_index=dataset_item.edge_index.clone(),
        y=dataset_item.y.clone() if dataset_item.y is not None else None,
    )


def signal_distribution(out):
    node_signal = out.abs().sum(dim=-1)
    total_signal = node_signal.sum()
    if total_signal <= 0:
        return torch.ones_like(node_signal) / node_signal.numel()
    return node_signal / total_signal


def get_resistances(graph, reference_node):
    resistances = {}
    for node in graph.nodes:
        if node == reference_node:
            resistances[node] = 0.0
        else:
            resistances[node] = nx.resistance_distance(
                graph, reference_node, node
            )
    return resistances


def process_graph_data(dataset_item, model_name, args, graph_idx):
    graph_data = prepare_graph(dataset_item)
    model = initialize_architecture(model_name, graph_data.x.size(-1), args)
    model.eval()

    graph = to_networkx(graph_data, to_undirected=True)
    distances = nx.floyd_warshall_numpy(graph)
    rows = []

    for sample_idx in range(args.num_sources):
        source = np.random.randint(0, len(graph))
        max_distance = distances[source].max()
        if max_distance == 0:
            continue

        x = torch.zeros_like(graph_data.x)
        x[source] = torch.randn_like(graph_data.x[source])
        x[source] = x[source].softmax(dim=-1)
        graph_data.x = x

        with torch.no_grad():
            if model_name == 'nsd':
                out = model(graph_data.x, graph_data.edge_index)
            else:
                out = model(graph_data)
            node_signal = signal_distribution(out).cpu().numpy()

        propagation = float(
            ((1 / max_distance) * (node_signal * distances[:, source])).mean()
        )
        total_effective_resistance = float(
            sum(get_resistances(graph, source).values())
        )

        rows.append({
            'graph_idx': graph_idx,
            'source_idx': sample_idx,
            'source_node': source,
            'model': model_name,
            'dataset': args.dataset,
            'layers': args.layers,
            'hidden_dim': args.hidden_dim,
            'effective_resistance': total_effective_resistance,
            'propagation': propagation,
        })

    return rows


def plot_results(results_df, output_path):
    fig, ax = plt.subplots(figsize=(9, 6))
    for model_name, model_df in results_df.groupby('model'):
        data = model_df[['effective_resistance', 'propagation']].dropna()
        if len(data) < 2:
            continue
        data = data.sort_values('effective_resistance')
        x = MinMaxScaler().fit_transform(
            data['effective_resistance'].to_numpy().reshape(-1, 1)
        ).flatten()
        y = MinMaxScaler().fit_transform(
            data['propagation'].to_numpy().reshape(-1, 1)
        ).flatten()
        plot_df = pd.DataFrame({'x': x, 'y': y})
        plot_df['smooth_y'] = plot_df['y'].ewm(halflife=2).mean()
        ax.plot(plot_df['x'], plot_df['smooth_y'], label=model_name)
        ax.scatter(plot_df['x'], plot_df['y'], s=10, alpha=0.25)

    ax.set_xlabel('Normalized total effective resistance')
    ax.set_ylabel('Normalized signal propagation')
    ax.set_title('Signal propagation vs effective resistance')
    ax.legend(title='Model')
    sns.despine(fig=fig, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_wandb_kwargs(args):
    entity = None if args.entity in {None, '', 'none', 'None'} else args.entity
    tags = [tag.strip() for tag in args.wandb_tags.split(',') if tag.strip()]
    tags.extend(['signal-propagation', args.dataset])
    return {
        'project': args.wandb_project,
        'entity': entity,
        'group': args.wandb_group,
        'name': args.wandb_run_name,
        'mode': args.wandb_mode,
        'tags': tags,
    }


def main():
    args = get_parser().parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    reset_wandb_env()

    repo = git.Repo(search_parent_directories=True)
    args.sha = repo.head.object.hexsha
    models = parse_models(args.models)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(config=vars(args), **build_wandb_kwargs(args))
    dataset = TUDataset(root=args.root, name=args.dataset)
    if args.max_graphs is not None:
        dataset = dataset[:args.max_graphs]

    rows = []
    for graph_idx, data in enumerate(dataset):
        for model_name in models:
            rows.extend(process_graph_data(data, model_name, args, graph_idx))

    results_df = pd.DataFrame(rows)
    csv_path = output_dir / f'{args.dataset}_signal_propagation.csv'
    png_path = output_dir / f'{args.dataset}_signal_propagation.png'
    results_df.to_csv(csv_path, index=False)
    plot_results(results_df, png_path)

    summary = (
        results_df
        .groupby('model', as_index=False)
        .agg(propagation_mean=('propagation', 'mean'),
             propagation_std=('propagation', 'std'),
             effective_resistance_mean=('effective_resistance', 'mean'),
             samples=('propagation', 'size'))
    )

    wandb.log({
        'results': wandb.Table(dataframe=results_df),
        'summary': wandb.Table(dataframe=summary),
        'signal_propagation_plot': wandb.Image(str(png_path)),
    })
    for row in summary.to_dict(orient='records'):
        wandb.summary[f"{row['model']}/propagation_mean"] = row['propagation_mean']
        wandb.summary[f"{row['model']}/effective_resistance_mean"] = row['effective_resistance_mean']
        wandb.summary[f"{row['model']}/samples"] = row['samples']
    wandb.finish()

    print(summary)
    print(f'Saved results to {csv_path}')
    print(f'Saved plot to {png_path}')


if __name__ == "__main__":
    main()
