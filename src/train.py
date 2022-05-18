"""Python file to train the model."""

"""

MITUSPTO = rxn smiles
LHS smiles      --> graph obj
graph obj       --> (MPNN) node features
node features   --> (Transformer) output sequence
output sequence --> (loss)

"""

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
from itertools import chain
from tqdm import tqdm
from torch_geometric.loader import DataLoader

import configs.train_cfg as cfg
from data.dataset import reaction_record_dataset
from utils.model_utils import load_models
from utils.model_utils import save_models



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
    train_loader = DataLoader(train_dataset, batch_size = cfg.BATCH_SIZE, shuffle = True)

    # ----- Load models
    model_enc, model_feedforward, model_dec = load_models(cfg)

    # ----- Get available device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_enc = model_enc.to(device)
    model_dec = model_dec.to(device)

    if model_feedforward:
        model_feedforward = model_feedforward.to(device)

    # ----- Load training settings
    all_params = chain(model_enc.parameters(), model_dec.parameters())
    optimizer = torch.optim.Adam(all_params, lr = cfg.LR, weight_decay = cfg.WEIGHT_DECAY)
    criterion = torch.nn.BCELoss() # TODO: seq2seq loss

    for epoch in range(cfg.EPOCHS):
        running_loss = 0.0
        for idx, train_batch in enumerate(train_loader):
            graph_batch, target_smiles_batch = train_batch
            print(type(graph_batch))
            print(graph_batch.size)
            graph_batch = graph_batch.to(device)
            target_smiles_batch = target_smiles_batch.to(device)
            optimizer.zero_grad()

            ## STEP 1: Standard Message passing operation on the graph
            # train_batch.x = 'BATCH' graph and train_batch.edge_matrix = 'BATCH' edge matrix
            atom_enc_features = model_enc(
                train_batch.x.float(), train_batch.edge_index, train_batch.edge_attr.float()
            )

            ## STEP 2: Forward pass on atom features using a feedforward network
            if model_feedforward:
                atom_mlp_features = model_feedforward(atom_enc_features)
            else:
                atom_mlp_features = atom_enc_features

            ## STEP 3: Generate the RHS text sequence from atom latent vectors
            print(atom_mlp_features.shape)
            break


            # loss = criterion(scores, train_batch.target.unsqueeze(1).float())

            # loss.backward()
            # optimizer.step()

            # # print statistics
            # running_loss += loss.item()
            # if idx % 100 == 99:    # print every 100 mini-batches
            #     save_models(cfg, model_mpnn, model_feedforward, model_scoring)
            #     print(f'At epoch: {epoch + 1}, minibatch: {idx + 1:5d} | running_loss: {running_loss}')
            #     running_loss = 0.0

    print('Finished Training')


if __name__ == '__main__':
    train()

