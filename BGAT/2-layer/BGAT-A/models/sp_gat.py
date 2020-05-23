#!/usr/local/bin/python
import numpy as np
import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
from utils import layers
from models.base_gattn import BaseGAttN

class SpGAT(BaseGAttN):
    def inference(inputs, nb_classes, nb_nodes, training, attn_drop, ffd_drop,
            bias_mat, adj_hop1_all_mat, adj_hop2_all_mat, adj_hop1_neig_mat,adj_hop2_neig_mat, N_hop1_all_mat,N_hop2_all_mat,
                  hid_units, n_heads, activation=tf.nn.elu,
            residual=False):

        attns = []
        for _ in range(n_heads[0]):
            attns.append(layers.sp_attn_head(inputs, adj_mat=bias_mat,
                                        adj_hop1_all_mat=adj_hop1_all_mat, adj_hop2_all_mat=adj_hop2_all_mat,
                                        adj_hop1_neig_mat=adj_hop1_neig_mat,adj_hop2_neig_mat=adj_hop2_neig_mat,
                                           N_hop1_all_mat=N_hop1_all_mat, N_hop2_all_mat=N_hop2_all_mat,
                                        out_sz=hid_units[0], activation=activation, nb_nodes=nb_nodes,
                                        in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        h_1 = tf.concat(attns, axis=-1)
        out = []
        for i in range(n_heads[-1]):
            out.append(layers.sp_attn_head(h_1, adj_mat=bias_mat,
                                            adj_hop1_all_mat=adj_hop1_all_mat, adj_hop2_all_mat=adj_hop2_all_mat,
                                            adj_hop1_neig_mat=adj_hop1_neig_mat,adj_hop2_neig_mat=adj_hop2_neig_mat,
                                           N_hop1_all_mat=N_hop1_all_mat, N_hop2_all_mat=N_hop2_all_mat,
                                            out_sz=nb_classes, activation=lambda x: x, nb_nodes=nb_nodes,
                                            in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        logits = tf.add_n(out) / n_heads[-1]


        bi = []
        bi.append(layers.Bilinear(inputs, adj_mat=bias_mat,
                                            adj_hop1_all_mat=adj_hop1_all_mat, adj_hop2_all_mat=adj_hop2_all_mat,
                                            adj_hop1_neig_mat=adj_hop1_neig_mat,adj_hop2_neig_mat=adj_hop2_neig_mat,
                                           N_hop1_all_mat=N_hop1_all_mat, N_hop2_all_mat=N_hop2_all_mat,
                                            out_sz=nb_classes, activation=lambda x: x, nb_nodes=nb_nodes,
                                            in_drop=ffd_drop, coef_drop=attn_drop, residual=False))
        bi_logits = tf.add_n(bi)

        output = (1. - FLAGS.alpha) * logits + FLAGS.alpha * bi_logits
        return output
