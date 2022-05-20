"""Python file for pytorch util functions."""

import os
import torch

from models.mpnn_models import GCN_2layer
from models.mpnn_models import GAT_2layer
from models.transformer_models import TransDecoder

def load_models(cfg):

    # model_mpnn = GAT_2layer(2, 32, 'train')
    model_mpnn = GCN_2layer(2, 32, 'train')
    model_enc = None # TODO
    model_dec = TransDecoder(
        cfg.EMBEDDING_DIM, cfg.NUM_HEADS, cfg.NUM_DECODER_LAYERS, cfg.VOCAB_SIZE, cfg.DEVICE
    )

    # if saved model exists, load it
    # if cfg.LOAD_MODEL_PATH and os.path.exists(cfg.LOAD_MODEL_PATH):
    #     model_weights = torch.load(cfg.LOAD_MODEL_PATH)
    #     model_mpnn.load_state_dict(model_weights['mpnn'])
    #     model_feedforward.load_state_dict(model_weights['feedforward'])
    #     model_scoring.load_state_dict(model_weights['scoring'])

    return model_mpnn, model_enc, model_dec

    pass

def save_models(cfg, model_mpnn, model_feedforward, model_scoring):

    # model_weights_dict = {
    #     'mpnn': model_mpnn.state_dict(),
    #     'feedforward': model_feedforward.state_dict(),
    #     'scoring': model_scoring.state_dict(),
    # }
    # torch.save(model_weights_dict, cfg.SAVE_MODEL_PATH)

    pass