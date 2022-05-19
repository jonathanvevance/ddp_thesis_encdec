"""Python file for pytorch utils."""

import torch
from torch.nn.utils.rnn import pad_sequence

# https://twitter.com/jeremyphoward/status/1185062637341593600?s=20&t=dE5aS3G8UxDERC_IUeuVCA
def groupby_pad_batch(graph_tensor_2d, graph_node_labels):
    """Return length-padded tensor and padding mask."""
    __, vals = torch.unique(graph_node_labels, return_counts = True)
    tensors_tuple = torch.split_with_sizes(graph_tensor_2d, tuple(vals))
    padded_tensor = pad_sequence(list(tensors_tuple), padding_value = -999) # (L x B x feat_size)
    pad_mask = torch.BoolTensor((padded_tensor[:, :, 0] == -999).cpu()).t() # (B x L)

    return padded_tensor, pad_mask
