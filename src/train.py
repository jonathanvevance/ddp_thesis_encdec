"""Python file to train the model."""
# https://www.analyticsvidhya.com/blog/2021/06/language-translation-with-transformer-in-python/
# https://hyugen-ai.medium.com/transformers-in-pytorch-from-scratch-for-nlp-beginners-ff3b3d922ef7

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
from itertools import chain
from torch_geometric.loader import DataLoader

import configs.train_cfg as cfg
from data.dataset import reaction_record_dataset
from utils.model_utils import load_models
from utils.model_utils import save_models
from utils.torch_utils import groupby_pad_batch
from utils.torch_utils import get_attention_mask

RAW_DATASET_PATH = 'data/raw/'

def train():
    """
    Train the neural network for pairwise-interaction matrix prediction stage.

    Implementation notes:
        pytorch-geometric batches graphs (datapoints) in the batch into a
        single large graph. Attributes attached to torch_geometric.data.Data
        objects are also batched in a specific way. I have taken advantage of
        this to attach the required attributes to Data objects.
    """

    # ----- Load dataset, dataloader
    train_dataset_filepath = os.path.join(RAW_DATASET_PATH, 'train.txt')
    train_dataset = reaction_record_dataset(
        dataset_filepath = train_dataset_filepath,
        mode = 'train',
    )
    cfg.VOCAB_SIZE = train_dataset.get_num_words() # adding vocab size to cfg object
    train_loader = DataLoader(train_dataset, batch_size = cfg.BATCH_SIZE, shuffle = True)

    # ----- Get available device
    cfg.DEVICE = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    ) # adding device to cfg object

    # ----- Load models
    model_mpnn, model_enc, model_dec, model_embedding = load_models(cfg)
    model_mpnn = model_mpnn.to(cfg.DEVICE)
    model_dec = model_dec.to(cfg.DEVICE)
    model_params = [model_mpnn.parameters(), model_dec.parameters()]

    if model_embedding:
        model_embedding = model_embedding.to(cfg.DEVICE)
        model_params = model_params + [model_embedding.parameters()]

    if model_enc:
        model_enc = model_enc.to(cfg.DEVICE)
        model_params = model_params + [model_enc.parameters()]

    # ----- Load training settings
    all_params = chain(*model_params)
    optimizer = torch.optim.Adam(all_params, lr = cfg.LR, weight_decay = cfg.WEIGHT_DECAY)
    criterion = torch.nn.CrossEntropyLoss(ignore_index = train_dataset.IGNORE_INDEX)

    # ----- Get target mask
    tgt_mask = get_attention_mask(train_dataset.get_longest_sentence())
    tgt_mask = tgt_mask.to(cfg.DEVICE) # to prevent cheating by looking ahead

    # ---- For debugging gradients
    torch.autograd.set_detect_anomaly(True)

    for epoch in range(cfg.EPOCHS):
        running_loss = 0.0
        for idx, train_batch in enumerate(train_loader):

            graph_batch, tgt_batch, tgt_pad_mask = train_batch
            graph_batch = graph_batch.to(cfg.DEVICE)
            tgt_batch = tgt_batch.to(cfg.DEVICE)
            tgt_pad_mask = tgt_pad_mask.to(cfg.DEVICE)

            optimizer.zero_grad()

            #! DEBUG
            if graph_batch.x.isnan().any():
                raise RuntimeError("Something wrong with input x")
            if graph_batch.edge_attr.isnan().any():
                raise RuntimeError("Something wrong with input edge")

            ## STEP 1: Get embeddings of graph features
            if model_embedding:
                graph_x, graph_edge_attr = model_embedding(graph_batch.x, graph_batch.edge_attr)
            else:
                graph_x = graph_batch.x.float()
                graph_edge_attr = graph_batch.edge_attr.float()

            #! DEBUG
            if graph_x.isnan().any():
                raise RuntimeError("Something wrong with embedding module - x")
            if graph_edge_attr.isnan().any():
                raise RuntimeError("Something wrong with embedding module - edge")

            ## STEP 2: Standard Message passing operation on the graph
            # train_batch.x = 'BATCH' graph and train_batch.edge_matrix = 'BATCH' edge matrix
            atom_enc_features = model_mpnn(graph_x, graph_batch.edge_index, graph_edge_attr)

            #! DEBUG
            if atom_enc_features.isnan().any():
                raise RuntimeError("Something wrong with mpnn module")

            ## Step 3: Reshape graph batch into Transformer compatible inputs
            atom_enc_features_batched, atom_enc_features_padding = groupby_pad_batch(
                atom_enc_features, graph_batch.batch
            )

            #! DEBUG
            if atom_enc_features_batched.isnan().any():
                raise RuntimeError("Something wrong with groupby module")

            ## STEP 4: Forward pass on atom features using a Transformer Encoder
            if model_enc:
                atom_enc_features_batched = model_enc(
                    atom_enc_features_batched, atom_enc_features_padding
                )

            #! DEBUG
            if tgt_batch.isnan().any():
                raise RuntimeError("Something wrong with input - tgt_batch")
            if tgt_pad_mask.isnan().any():
                raise RuntimeError("Something wrong with input - tgt_pad_mask")
            if atom_enc_features_padding.isnan().any():
                raise RuntimeError("Something wrong with input - atom_enc_features_padding")

            ## STEP 5: Generate the RHS text sequence logits from atom latent vectors
            logits = model_dec(
                tgt_batch, tgt_pad_mask, tgt_mask, atom_enc_features_batched, atom_enc_features_padding
            )

            #! DEBUG
            if logits.isnan().any():
                raise RuntimeError("Something wrong with input - logits")

            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_batch.reshape(-1))
            loss.backward()

            if cfg.GRAD_CLIP: # Gradient clipping
                torch.nn.utils.clip_grad_norm_(all_params, max_norm = cfg.GRAD_CLIP_MAX_NORM)
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if idx % 100 == 99:    # print every 100 mini-batches
                save_models(cfg, model_mpnn, model_enc, model_dec, model_embedding)
                print(f'At epoch: {epoch + 1}, minibatch: {idx + 1:5d} | running_loss: {running_loss}')
                running_loss = 0.0

    print('Finished Training')


if __name__ == '__main__':
    train()

