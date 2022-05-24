"""Python file for pytorch util functions."""

import os
import torch

from models.mpnn_models import GCN_2layer
from models.mpnn_models import GAT_2layer
from models.transformer_models import TransDecoder

def load_models(cfg):

    # model_mpnn = GAT_2layer(2, cfg.EMBEDDING_DIM, 'train')
    model_mpnn = GCN_2layer(2, cfg.EMBEDDING_DIM, 'train')
    model_enc = None # TODO
    model_dec = TransDecoder(
        cfg.EMBEDDING_DIM, cfg.NUM_HEADS, cfg.NUM_DECODER_LAYERS, cfg.VOCAB_SIZE, cfg.DEVICE
    )

    # if saved model exists, load it
    if cfg.LOAD_MODEL_PATH and os.path.exists(cfg.LOAD_MODEL_PATH):
        model_weights = torch.load(cfg.LOAD_MODEL_PATH)
        model_mpnn.load_state_dict(model_weights['mpnn'])
        if model_enc:
            model_enc.load_state_dict(model_weights['enc'])
        model_dec.load_state_dict(model_weights['dec'])

    return model_mpnn, model_enc, model_dec

def save_models(cfg, model_mpnn, model_enc, model_dec):

    model_weights_dict = {
        'mpnn': model_mpnn.state_dict(),
        'enc': model_enc.state_dict() if model_enc else None,
        'dec': model_dec.state_dict(),
    }
    torch.save(model_weights_dict, cfg.SAVE_MODEL_PATH)
