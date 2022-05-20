"""Python file with dataset classes."""

import os
import torch
import torch.nn as nn
from tqdm import tqdm
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.data import Dataset

from utils.rdkit_utils import get_pyg_graph_requirements
from utils.rdkit_utils import remove_atom_maps_from_smi
from utils.vocab_utils import Vocabulary
from utils.vocab_utils import smi_tokenizer
from utils.vocab_utils import PAD_INDEX

PROCESSED_DATASET_LOC = 'data/processed/'

class reaction_record:
    def __init__(self, reaction_smiles, vocabulary):

        lhs_smiles, rhs_smiles = reaction_smiles.split(">>")
        self.lhs_mol = Chem.MolFromSmiles(lhs_smiles)

        pyg_requirements = get_pyg_graph_requirements(self.lhs_mol)
        self.pyg_data = Data(
            x = torch.tensor(pyg_requirements['x']),
            edge_index = torch.tensor(pyg_requirements['edge_index']),
            edge_attr = torch.tensor(pyg_requirements['edge_attr']),
        )

        rhs_wordidx_list = []
        rhs_smiles = rhs_smiles.split(' ')[0] # HACK to remove junk chars from end of rhs
        rhs_smi_sentence = smi_tokenizer(remove_atom_maps_from_smi(rhs_smiles))
        for word in rhs_smi_sentence.split(' '):
            rhs_wordidx_list.append(vocabulary.to_index(word))

        const_padder = nn.ConstantPad1d(
            (0, vocabulary.longest_sentence - len(rhs_wordidx_list)), PAD_INDEX
        )
        self.tgt_wordidx_tensor = const_padder(torch.Tensor(rhs_wordidx_list))
        self.tgt_wordidx_pad_mask = torch.BoolTensor((self.tgt_wordidx_tensor == PAD_INDEX).cpu()).t()

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
        self.vocab = None
        self.dataset_filepath = dataset_filepath
        self.processed_mode_dir = os.path.join(PROCESSED_DATASET_LOC, self.mode)
        self.processed_filepaths = []

        self.process_reactions()

    def get_smiles_vocab(self):
        """Get smiles vocabulary and save to disk (if unsaved)."""
        vocab_savepath = os.path.join(PROCESSED_DATASET_LOC, 'vocab.pt')
        if os.path.exists(vocab_savepath):
            self.vocab = torch.load(vocab_savepath)
            return

        self.vocab = Vocabulary()
        num_rxns = sum(1 for line in open(self.dataset_filepath, "r"))
        with open(self.dataset_filepath, "r") as train_dataset:
            for rxn_num, reaction_smiles in enumerate(tqdm(
                train_dataset, desc = f"Getting vocabulary from train smiles", total = num_rxns
            )):

                lhs_smiles, rhs_smiles = reaction_smiles.split(">>")
                lhs_smi_sentence = smi_tokenizer(remove_atom_maps_from_smi(lhs_smiles))
                rhs_smiles = rhs_smiles.split(' ')[0] # HACK to remove junk chars from end of rhs
                rhs_smi_sentence = smi_tokenizer(remove_atom_maps_from_smi(rhs_smiles))
                self.vocab.add_sentence(lhs_smi_sentence)
                self.vocab.add_sentence(rhs_smi_sentence)

        torch.save(self.vocab, vocab_savepath)

    def process_reactions(self):
        """Process each reaction in the dataset."""

        if not os.path.exists(self.processed_mode_dir):
            os.makedirs(self.processed_mode_dir)

        if self.mode == 'train':
            self.get_smiles_vocab()

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

                reaction = reaction_record(reaction_smiles, self.vocab)

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

        return (
            reaction_data.pyg_data,
            reaction_data.tgt_wordidx_tensor,
            reaction_data.tgt_wordidx_pad_mask
        )
