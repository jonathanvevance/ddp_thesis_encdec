"""Python file for pytorch util functions."""

import os
import torch

from models.mpnn_models import GCN_2layer
from models.mpnn_models import GAT_2layer
from models.transformer_models import TransDecoder
from models.embedding_models import FeatureEmbedding

def set_train_mode(models):
    for model in models:
        model.train()

def set_eval_mode(models):
    for model in models:
        model.eval()

def load_models(cfg):

    if cfg.USE_LHS_EMBEDDING:
        model_embedding = FeatureEmbedding(
            cfg.LHS_EMBEDDING_DIM, cfg.UNIQUE_ATOMS, cfg.UNIQUE_CHARGES, cfg.UNIQUE_BONDS
        )
    else:
        model_embedding = None

    if cfg.USE_TRANS_ENCODER:
        model_enc = None # TODO
        pass
    else:
        model_enc = None

    model_mpnn = GCN_2layer(cfg.MPNN_FEATURES_DIM, cfg.RHS_EMBEDDING_DIM, 'train')
    model_dec = TransDecoder(
        cfg.RHS_EMBEDDING_DIM, cfg.NUM_HEADS, cfg.NUM_DECODER_LAYERS, cfg.VOCAB_SIZE, cfg.DEVICE
    )

    # if saved model exists, load it
    if cfg.LOAD_MODEL_PATH and os.path.exists(cfg.LOAD_MODEL_PATH):

        model_weights = torch.load(cfg.LOAD_MODEL_PATH)
        model_mpnn.load_state_dict(model_weights['mpnn'])
        model_dec.load_state_dict(model_weights['dec'])

        if model_enc and 'enc' in model_weights:
            model_enc.load_state_dict(model_weights['enc'])

        if model_embedding and 'embedding' in model_weights:
            model_embedding.load_state_dict(model_weights['embedding'])

    return model_mpnn, model_enc, model_dec, model_embedding

def save_models(cfg, model_mpnn, model_enc, model_dec, model_embedding):

    model_weights_dict = {
        'mpnn': model_mpnn.state_dict(),
        'enc': model_enc.state_dict() if model_enc else None,
        'dec': model_dec.state_dict(),
        'embedding': model_embedding.state_dict() if model_embedding else None,
    }
    torch.save(model_weights_dict, cfg.SAVE_MODEL_PATH)
