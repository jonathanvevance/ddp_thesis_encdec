"""Python file with dataset classes."""

import os
import torch
from tqdm import tqdm
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.data import Dataset

from rdkit_helpers.features import get_pyg_graph_requirements

PROCESSED_DATASET_LOC = 'data/processed/'

class reaction_record:
    def __init__(self, reaction_smiles):

        lhs_smiles, rhs_smiles = reaction_smiles.split(">>")
        self.rhs_smiles = rhs_smiles
        self.lhs_mol = Chem.MolFromSmiles(lhs_smiles)

        pyg_requirements = get_pyg_graph_requirements(self.lhs_mol)
        self.pyg_data = Data(
            x = torch.tensor(pyg_requirements['x']),
            edge_index = torch.tensor(pyg_requirements['edge_index']),
            edge_attr = torch.tensor(pyg_requirements['edge_attr']),
        )

class reaction_record_dataset(Dataset):

    def __init__(
        self,
        dataset_filepath,
        mode='train',
        transform = None,
        pre_transform = None,
        pre_filter = None
    ):

        # https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html
        super().__init__(
            None,
            transform,
            pre_transform,
            pre_filter
        ) # None to skip downloading (see FAQ)

        self.mode = mode
        self.dataset_filepath = dataset_filepath
        self.processed_mode_dir = os.path.join(PROCESSED_DATASET_LOC, self.mode)
        self.processed_filepaths = []

        self.process_reactions()

    def process_reactions(self):
        """Process each reaction in the dataset."""

        if not os.path.exists(self.processed_mode_dir):
            os.makedirs(self.processed_mode_dir)

        reaction_files = os.listdir(self.processed_mode_dir)
        if len(reaction_files):
            start_from = max(int(reaction_file[4:-3]) for reaction_file in reaction_files)
        else:
            start_from = -1

        num_rxns = sum(1 for line in open(self.dataset_filepath, "r"))

        with open(self.dataset_filepath, "r") as train_dataset:
            for rxn_num, reaction_smiles in enumerate(tqdm(
                train_dataset, desc = f"Preparing {self.mode} reactions", total = num_rxns
            )):

                if rxn_num == 500: return # TODO: remove

                processed_filepath = os.path.join(self.processed_mode_dir, f'rxn_{rxn_num}.pt')
                if rxn_num < start_from + 1:
                    if os.path.exists(processed_filepath):
                        self.processed_filepaths.append(processed_filepath)
                    continue

                reaction = reaction_record(reaction_smiles)

                if self.pre_filter is not None and not self.pre_filter(reaction):
                    continue

                if self.pre_transform is not None:
                    reaction = self.pre_transform(reaction)

                torch.save(reaction, processed_filepath)
                self.processed_filepaths.append(processed_filepath)

    def len(self):
        """Get length of reaction dataset."""
        return len(self.processed_filepaths)

    def get(self, idx):
        """Get data point for given reaction-idx."""

        processed_filepath = self.processed_filepaths[idx]
        reaction_data = torch.load(processed_filepath) # load graph

        return reaction_data.pyg_data, reaction_data.rhs_smiles
        # TODO: return rhs smiles also