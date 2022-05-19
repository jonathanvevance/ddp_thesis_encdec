"""Python file for pytorch utils."""

import torch
from torch.nn.utils.rnn import pad_sequence

# https://twitter.com/jeremyphoward/status/1185062637341593600?s=20&t=dE5aS3G8UxDERC_IUeuVCA
def groupby_pad_batch(graph_tensor_2d, graph_node_labels):
    """Pad sequences to ."""
    __, vals = torch.unique(graph_node_labels, return_counts = True)
    tensors_tuple = torch.split_with_sizes(graph_tensor_2d, tuple(vals))
    return pad_sequence(list(tensors_tuple))
