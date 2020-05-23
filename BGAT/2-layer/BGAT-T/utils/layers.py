#!/usr/local/bin/python
import numpy as np
import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
conv1d = tf.layers.conv1d


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

def BILinear_pooling(adj_,XW):

    #step1 sum_squared
    sum = dot(adj_, XW, True)

    #step2 squared_sum
    squared = tf.multiply(sum, XW)

    #step3
    new_embedding = squared

    return new_embedding

def Bilinear(seq, out_sz, adj_mat, adj_hop1_all_mat, adj_hop2_all_mat, adj_hop1_neig_mat,adj_hop2_neig_mat,
                 N_hop1_target_mat,N_hop2_target_mat,activation, nb_nodes, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('bilinear'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)
        if coef_drop != 0.0:
            adj_hop1_neig_mat = tf.SparseTensor(indices=adj_hop1_neig_mat.indices,
                                    values=tf.nn.dropout(adj_hop1_neig_mat.values, 1.0 - coef_drop),
                                    dense_shape=adj_hop1_neig_mat.dense_shape)
            adj_hop2_neig_mat = tf.SparseTensor(indices=adj_hop2_neig_mat.indices,
                                               values=tf.nn.dropout(adj_hop2_neig_mat.values, 1.0 - coef_drop),
                                               dense_shape=adj_hop2_neig_mat.dense_shape)

        seq_fts = tf.squeeze(seq_fts)  ###HW
        out_bi1 = dot(N_hop1_target_mat, BILinear_pooling(adj_hop1_neig_mat, seq_fts), True)
        out_bi2 = dot(N_hop2_target_mat, BILinear_pooling(adj_hop2_neig_mat, seq_fts), True)
        out_bi = (1 - FLAGS.beta) * out_bi1 + FLAGS.beta * out_bi2


        vals_out = tf.expand_dims(out_bi, axis=0)
        vals_out.set_shape([1, nb_nodes, out_sz])

        return activation(vals_out)  # activation


def sp_attn_head(seq, out_sz, adj_mat, adj_hop1_all_mat, adj_hop2_all_mat, adj_hop1_neig_mat,adj_hop2_neig_mat,
                 N_hop1_target_mat,N_hop2_target_mat,activation, nb_nodes, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('sp_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        ###this is the first layer of GAT
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)

        f_1 = tf.reshape(f_1, (nb_nodes, 1))
        f_2 = tf.reshape(f_2, (nb_nodes, 1))

        f_1 = adj_mat * f_1
        f_2 = adj_mat * tf.transpose(f_2, [1, 0])

        logits = tf.sparse_add(f_1, f_2)
        lrelu = tf.SparseTensor(indices=logits.indices,
                                values=tf.nn.leaky_relu(logits.values),
                                dense_shape=logits.dense_shape)
        coefs = tf.sparse_softmax(lrelu)

        if coef_drop != 0.0:
            coefs = tf.SparseTensor(indices=coefs.indices,
                                    values=tf.nn.dropout(coefs.values, 1.0 - coef_drop),
                                    dense_shape=coefs.dense_shape)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)


        coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])
        seq_fts = tf.squeeze(seq_fts)
        vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)
        vals = tf.expand_dims(vals, axis=0)
        vals.set_shape([1, nb_nodes, out_sz])
        ret = tf.contrib.layers.bias_add(vals)

        return activation(ret)  # activation