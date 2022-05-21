"""Python file for pytorch utils."""

import torch
from torch.nn.utils.rnn import pad_sequence

# https://twitter.com/jeremyphoward/status/1185062637341593600?s=20&t=dE5aS3G8UxDERC_IUeuVCA
# https://stackoverflow.com/questions/61688282/how-to-get-padding-mask-from-input-ids
# https://stackoverflow.com/questions/62399243/transformerencoder-with-a-padding-mask
# https://stackoverflow.com/questions/62170439/difference-between-src-mask-and-src-key-padding-mask

def groupby_pad_batch(graph_tensor_2d, graph_node_labels):
    """Return length-padded tensor and padding mask."""
    __, vals = torch.unique(graph_node_labels, return_counts = True)
    tensors_tuple = torch.split_with_sizes(graph_tensor_2d, tuple(vals))
    padded_tensor = pad_sequence(list(tensors_tuple), padding_value = 0) # (L x B x feat_size)
    pad_mask = torch.BoolTensor((padded_tensor[:, :, 0] == 0).cpu())\
                    .t().to(padded_tensor.device) # (B x L)
    return padded_tensor, pad_mask
