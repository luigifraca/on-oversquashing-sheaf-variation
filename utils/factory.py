import torch
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
        super(Net, self).__init__()

        name2arch = {'gcn': GCNConv, 'sage': SAGEConv, 'gat': GATConv, 'gin': GINConv}

        module_list = []
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
          if arch != 'gin':
            self.convs.append(
                Sequential(name2arch[arch](dim_h if i != 0 else dataset.num_node_features, dim_h, bias=False, root_weight=False),
                           Identity(dim_h),
                          ReLU()
                          )
            )

          elif arch == 'gin':
            self.convs.append(GINConv( Sequential(
                                                  Linear(dim_h if i != 0 else dataset.num_node_features, dim_h, bias=False),
                                                  Identity(dim_h), ReLU(),
                                                  Linear(dim_h, dim_h, bias=False),
                                                  ReLU()
                                                 )
                                     )
                             )

        self.arch = arch

    def forward(self, G):
        h, edge_index = G.x, G.edge_index
        for conv in self.convs:
          if self.arch == 'gin':
            h = conv(h, edge_index)
          else:
            for op in conv:
              try:
                h = op(h, edge_index)
              except:
                h = op(h)
        return h
