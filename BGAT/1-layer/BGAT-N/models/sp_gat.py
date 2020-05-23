#!/usr/local/bin/python
import numpy as np
import tensorflow as tf

from utils import layers
from models.base_gattn import BaseGAttN

class SpGAT(BaseGAttN):
    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
            bias_mat, adj_all_mat, adj_neig_mat, N_neig_mat, hid_units, n_heads, activation=tf.nn.elu,
            residual=False):
        out = []
        for i in range(n_heads[-1]):
            out.append(layers.sp_attn_head(inputs, adj_mat=bias_mat,adj_all_mat=adj_all_mat, adj_neig_mat=adj_neig_mat, N_neig_mat=N_neig_mat,
                out_sz=nb_classes, activation=lambda x: x, nb_nodes=nb_nodes,
                in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        logits = tf.add_n(out) / n_heads[-1]
    
        return logits
