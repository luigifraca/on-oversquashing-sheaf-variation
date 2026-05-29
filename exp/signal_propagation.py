import argparse
import csv
import fcntl
import hashlib
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
    parser.add_argument('--signal_dim', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=5)
    parser.add_argument('--max_graphs', type=int, default=None)
    parser.add_argument('--num_sources', type=int, default=10)
    parser.add_argument('--seed', type=int, default=808)
    parser.add_argument('--torch_num_threads', type=int, default=None)
    parser.add_argument('--torch_num_interop_threads', type=int, default=None)
    parser.add_argument('--output_dir', type=str,
                        default='signal_propagation_results')
    parser.add_argument('--cache_graph_metrics', type=str2bool, default=False)
    parser.add_argument('--graph_metrics_cache_dir', type=str,
                        default='data/signal_metric_cache')

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
    parser.add_argument('--results_file', type=str, default=None)
    parser.add_argument('--raw_results_file', type=str, default=None)
    parser.add_argument('--experiment_label', type=str, default=None)
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


def prepare_graph(dataset_item, signal_dim):
    x = torch.zeros((dataset_item.num_nodes, signal_dim), dtype=torch.float32)
    return Data(
        x=x,
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


def total_effective_resistances(graph):
    if not nx.is_connected(graph):
        return np.full(graph.number_of_nodes(), np.inf)

    laplacian = nx.laplacian_matrix(graph).toarray().astype(np.float64)
    laplacian_pinv = np.linalg.pinv(laplacian, hermitian=True)
    trace = np.trace(laplacian_pinv)
    row_sums = laplacian_pinv.sum(axis=1)
    totals = graph.number_of_nodes() * np.diag(laplacian_pinv) + trace
    totals -= 2 * row_sums
    return np.maximum(totals, 0.0)


def graph_metric_cache_path(dataset_item, args, graph_idx):
    edge_index = dataset_item.edge_index.cpu().contiguous()
    digest = hashlib.sha1()
    digest.update(str(dataset_item.num_nodes).encode('utf-8'))
    digest.update(edge_index.numpy().tobytes())
    graph_hash = digest.hexdigest()[:16]
    name = (
        f'{args.dataset}_g{graph_idx}_s{args.seed}_n{args.num_sources}_'
        f'{graph_hash}.pt'
    )
    return Path(args.graph_metrics_cache_dir) / name


def torch_load(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def save_atomic(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def build_graph_metrics(dataset_item, args, graph_idx):
    graph_data = prepare_graph(dataset_item, args.signal_dim)
    graph = to_networkx(graph_data, to_undirected=True)
    distances = np.asarray(nx.floyd_warshall_numpy(graph), dtype=np.float64)
    resistance_totals = total_effective_resistances(graph)
    rng = np.random.default_rng(args.seed + graph_idx * 1_000_003)

    source_indices = []
    sources = []
    distance_columns = []
    max_distances = []
    effective_resistances = []
    for sample_idx in range(args.num_sources):
        source = int(rng.integers(0, len(graph)))
        max_distance = float(distances[source].max())
        if max_distance == 0 or not np.isfinite(max_distance):
            continue

        source_indices.append(sample_idx)
        sources.append(source)
        distance_columns.append(distances[:, source])
        max_distances.append(max_distance)
        effective_resistances.append(float(resistance_totals[source]))

    return {
        'source_indices': source_indices,
        'sources': sources,
        'distance_columns': np.asarray(distance_columns, dtype=np.float64),
        'max_distances': np.asarray(max_distances, dtype=np.float64),
        'effective_resistances': np.asarray(effective_resistances,
                                            dtype=np.float64),
    }


def get_graph_metrics(dataset_item, args, graph_idx):
    if not args.cache_graph_metrics:
        return build_graph_metrics(dataset_item, args, graph_idx)

    cache_path = graph_metric_cache_path(dataset_item, args, graph_idx)
    if cache_path.exists():
        return torch_load(cache_path)

    metrics = build_graph_metrics(dataset_item, args, graph_idx)
    save_atomic(metrics, cache_path)
    return metrics


def process_graph_data(dataset_item, model_name, args, graph_idx,
                       graph_metrics):
    graph_data = prepare_graph(dataset_item, args.signal_dim)
    model = initialize_architecture(model_name, args.signal_dim, args)
    model.eval()

    rows = []

    for metric_idx, sample_idx in enumerate(graph_metrics['source_indices']):
        source = graph_metrics['sources'][metric_idx]
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

        max_distance = graph_metrics['max_distances'][metric_idx]
        distance_column = graph_metrics['distance_columns'][metric_idx]
        propagation = float(
            ((1 / max_distance) * (node_signal * distance_column)).mean()
        )

        rows.append({
            'graph_idx': graph_idx,
            'source_idx': sample_idx,
            'source_node': source,
            'model': model_name,
            'dataset': args.dataset,
            'layers': args.layers,
            'signal_dim': args.signal_dim,
            'hidden_dim': args.hidden_dim,
            'effective_resistance': (
                graph_metrics['effective_resistances'][metric_idx]
            ),
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
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, title='Model')
    sns.despine(fig=fig, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def append_summary(results_file, summary_df, args, sha):
    if results_file is None:
        return

    path = Path(results_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        'task',
        'experiment_label',
        'dataset',
        'model',
        'sheaf_variant',
        'sheaf_normalize_output',
        'sheaf_add_self_loops',
        'stalk_dim',
        'hidden_dim',
        'signal_dim',
        'layers',
        'seed',
        'max_graphs',
        'num_sources',
        'propagation_mean',
        'propagation_std',
        'effective_resistance_mean',
        'samples',
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
        for row in summary_df.to_dict(orient='records'):
            writer.writerow({
                'task': 'signal-propagation',
                'experiment_label': args.experiment_label,
                'dataset': args.dataset,
                'model': row['model'],
                'sheaf_variant': args.sheaf_variant if row['model'] == 'nsd' else '',
                'sheaf_normalize_output': args.sheaf_normalize_output if row['model'] == 'nsd' else '',
                'sheaf_add_self_loops': args.sheaf_add_self_loops if row['model'] == 'nsd' else '',
                'stalk_dim': args.stalk_dim if row['model'] == 'nsd' else '',
                'hidden_dim': args.hidden_dim,
                'signal_dim': args.signal_dim,
                'layers': args.layers,
                'seed': args.seed,
                'max_graphs': args.max_graphs,
                'num_sources': args.num_sources,
                'propagation_mean': row['propagation_mean'],
                'propagation_std': row['propagation_std'],
                'effective_resistance_mean': row['effective_resistance_mean'],
                'samples': row['samples'],
                'sha': sha,
                'wandb_project': args.wandb_project,
                'wandb_group': args.wandb_group,
                'wandb_run_name': args.wandb_run_name,
            })
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


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
    if args.torch_num_threads is not None:
        torch.set_num_threads(args.torch_num_threads)
    if args.torch_num_interop_threads is not None:
        torch.set_num_interop_threads(args.torch_num_interop_threads)

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
        graph_metrics = get_graph_metrics(data, args, graph_idx)
        for model_name in models:
            rows.extend(
                process_graph_data(data, model_name, args, graph_idx,
                                   graph_metrics)
            )

    results_df = pd.DataFrame(rows)
    run_slug = args.wandb_run_name or f'{args.dataset}_signal_propagation'
    csv_path = Path(args.raw_results_file) if args.raw_results_file else output_dir / f'{run_slug}_raw.csv'
    png_path = output_dir / f'{args.dataset}_signal_propagation.png'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
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
    append_summary(args.results_file, summary, args, args.sha)
    wandb.finish()

    print(summary)
    print(f'Saved results to {csv_path}')
    print(f'Saved plot to {png_path}')


if __name__ == "__main__":
    main()
