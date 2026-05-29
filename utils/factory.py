import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv
from torch_geometric.nn.models import GIN, GCN, GraphSAGE, GAT

from data.ring_transfer import generate_tree_transfer_graph_dataset
from data.ring_transfer import generate_ring_transfer_graph_dataset
from data.ring_transfer import generate_lollipop_transfer_graph_dataset



def build_model(args):
	assert args.model in ['gin', 'gcn', 'gat', 'sage', 'nsd'], ValueError(f'Unknown model {args.model}')
	assert args.input_dim != None, ValueError(f'Invalid input dim')
	assert args.hidden_dim != None, ValueError(f'Invalid hidden dim')
	assert args.output_dim != None, ValueError(f'Invalid output dim')
	assert args.mpnn_layers != None, ValueError(f'Invalid number of mpnn layer')
	assert args.norm != None, ValueError(f'Invalid normalisation')

	if args.model == 'nsd':
		try:
			from sheaf_mpnn import NSDModel, NSDVariant
		except ImportError as exc:
			raise ImportError(
				"Model 'nsd' requires sheaf_mpnn. Use the Python 3.13 "
				"environment where /Users/luigifracassetti/projects/"
				"sheaf_mpnn_study is installed in editable mode."
			) from exc

		variant = NSDVariant[args.sheaf_variant.upper()]
		return NSDModel(
			in_channels=args.input_dim,
			out_channels=args.output_dim,
			stalk_dim=args.stalk_dim,
			hidden_dim=args.hidden_dim,
			num_layers=args.mpnn_layers,
			variant=variant,
			alpha=args.sheaf_alpha,
			add_self_loops=args.sheaf_add_self_loops,
			orth_strategy=args.sheaf_orth_strategy,
			rank=args.sheaf_rank,
			input_dropout=args.input_dropout,
			dropout=args.dropout,
			normalize_output=args.sheaf_normalize_output,
			jknet=args.sheaf_jknet,
		)

	models = {
	        'gin' : GIN,
	        'gcn' : GCN,
	        'gat' : GAT,
	        'sage' : GraphSAGE,
	     }

	params = {
	      'in_channels':args.input_dim,
	      'hidden_channels':args.hidden_dim,
	      'out_channels':args.output_dim,
	      'num_layers':args.mpnn_layers,
	      'norm':args.norm
	     }
	return models[args.model](**params)


def build_dataset(args):
    assert args.dataset in ['TREE', 'RING', 'LOLLIPOP'], ValueError(f'Unknown dataset {args.dataset}')

    dataset_factory = {
        'TREE': generate_tree_transfer_graph_dataset,
        'RING': generate_ring_transfer_graph_dataset,
        'LOLLIPOP': generate_lollipop_transfer_graph_dataset
    }

    dataset_configs = {
        'depth': args.synthetic_size,
        'nodes': args.synthetic_size,
        'classes': args.num_class,
        'samples': args.synth_train_size + args.synth_test_size,
        'arity': args.arity,
        'add_crosses': int(args.add_crosses)
    }

    return dataset_factory[args.dataset](**dataset_configs)




class NetFactory(torch.nn.Module):
    def __init__(self, arch, num_layers, dim_h):
        super().__init__()
        if arch not in {'gcn', 'sage', 'gat', 'gin'}:
            raise ValueError(f'Unknown architecture {arch}')
        if num_layers <= 0:
            raise ValueError('num_layers must be positive')

        self.arch = arch
        self.num_layers = num_layers
        self.dim_h = dim_h
        self.convs = torch.nn.ModuleList()

    def _build(self, in_channels, device):
        for layer in range(self.num_layers):
            layer_in = in_channels if layer == 0 else self.dim_h
            if self.arch == 'gcn':
                conv = GCNConv(layer_in, self.dim_h)
            elif self.arch == 'sage':
                conv = SAGEConv(layer_in, self.dim_h)
            elif self.arch == 'gat':
                conv = GATConv(layer_in, self.dim_h, heads=1, concat=False)
            else:
                mlp = nn.Sequential(
                    nn.Linear(layer_in, self.dim_h),
                    nn.ReLU(),
                    nn.Linear(self.dim_h, self.dim_h),
                )
                conv = GINConv(mlp)
            self.convs.append(conv.to(device))

    def forward(self, G):
        h, edge_index = G.x, G.edge_index
        if len(self.convs) == 0:
            self._build(h.size(-1), h.device)

        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
        return h
