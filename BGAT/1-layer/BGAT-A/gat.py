#!/usr/local/bin/python
import time
import numpy as np
import tensorflow as tf
from models.sp_gat import SpGAT
from utils import process

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)


# training params
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'citeseer', 'Dataset string.')
flags.DEFINE_string('model', 'SpGAT', 'Model string.')
flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 2000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('head', 1, 'Number of atten heads.')
flags.DEFINE_integer('patience', 100, 'early stop #epoch')
flags.DEFINE_float('feadrop', 0.0, 'feature Dropout rate (1 - keep probability).')
flags.DEFINE_float('attdrop', 0.6, 'attention Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('alpha', 0.9, 'alpha balancing BGCN')
residual = False
nonlinearity = lambda x: x
model = SpGAT

checkpt_file = '/pre_trained/'+FLAGS.dataset+'/'+'mod_'+FLAGS.dataset+'.ckpt'
#some preprocess
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = process.load_data(FLAGS.dataset)
features, spars = process.preprocess_features(features)
nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = y_train.shape[1]
features = features[np.newaxis]
y_train = y_train[np.newaxis]
y_val = y_val[np.newaxis]
y_test = y_test[np.newaxis]
train_mask = train_mask[np.newaxis]
val_mask = val_mask[np.newaxis]
test_mask = test_mask[np.newaxis]
biases = process.preprocess_adj_bias(adj)
adj_all, adj_neig, N_all = process.preprocess_bilinear(adj)

t_rep_s = time.time()

ftr_in = tf.placeholder(dtype=tf.float32, shape=(1, nb_nodes, ft_size))
bias_in = tf.sparse_placeholder(dtype=tf.float32)
adj_all_in = tf.sparse_placeholder(dtype=tf.float32)
adj_neig_in = tf.sparse_placeholder(dtype=tf.float32)
N_all_in = tf.sparse_placeholder(dtype=tf.float32)
lbl_in = tf.placeholder(dtype=tf.int32, shape=(1, nb_nodes, nb_classes))
msk_in = tf.placeholder(dtype=tf.int32, shape=(1, nb_nodes))
attn_drop = tf.placeholder(dtype=tf.float32, shape=())
ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
is_train = tf.placeholder(dtype=tf.bool, shape=())

logits = model.inference(ftr_in, nb_classes, nb_nodes, is_train,
                            attn_drop, ffd_drop,
                            bias_mat=bias_in,
                            adj_all_mat = adj_all_in,
                            adj_neig_mat = adj_neig_in,
                            N_all_mat = N_all_in,
                            hid_units=[FLAGS.hidden1], n_heads=[FLAGS.head],
                            residual=residual, activation=nonlinearity)
log_resh = tf.reshape(logits, [-1, nb_classes])
lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
msk_resh = tf.reshape(msk_in, [-1])
loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)

train_op = model.training(loss, FLAGS.learning_rate, FLAGS.weight_decay)

saver = tf.train.Saver()

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

vlss_mn = np.inf
vacc_mx = 0.0
curr_step = 0

sess = tf.Session()
sess.run(init_op)

for epoch in range(FLAGS.epochs):

    #train
    t = time.time()
    bbias = biases
    _, loss_value_tr, acc_tr = sess.run([train_op, loss, accuracy],
        feed_dict={
            ftr_in: features[0:1],
            bias_in: bbias,
            adj_all_in : adj_all,
            adj_neig_in : adj_neig,
            N_all_in : N_all,
            lbl_in: y_train[0:1],
            msk_in: train_mask[0:1],
            is_train: True,
            attn_drop: FLAGS.feadrop, ffd_drop: FLAGS.attdrop})

    #validation
    bbias = biases
    loss_value_vl, acc_vl = sess.run([loss, accuracy],
        feed_dict={
            ftr_in: features[0:1],
            bias_in: bbias,
            adj_all_in : adj_all,
            adj_neig_in : adj_neig,
            N_all_in : N_all,
            lbl_in: y_val[0:1],
            msk_in: val_mask[0:1],
            is_train: False,
            attn_drop: 0.0, ffd_drop: 0.0})

    #test
    bbias = biases
    loss_value_ts, acc_ts = sess.run([loss, accuracy],
        feed_dict={
        ftr_in: features[0:1],
        bias_in: bbias,
        adj_all_in : adj_all,
        adj_neig_in : adj_neig,
        N_all_in : N_all,
        lbl_in: y_test[0:1],
        msk_in: test_mask[0:1],
        is_train: False,
        attn_drop: 0.0, ffd_drop: 0.0})

    print("Epoch:", '%04d' % (epoch + 1),
          "train_loss=", "{:.5f}".format(loss_value_tr), "train_acc=", "{:.4f}".format(acc_tr),
          "val_loss=", "{:.5f}".format(loss_value_vl), "val_acc=", "{:.4f}".format(acc_vl),
          "tst_loss=", "{:.5f}".format(loss_value_ts), "tst_acc=", "{:.4f}".format(acc_ts),
          "time=", "{:.3f}".format(time.time() - t))

    if acc_vl >= vacc_mx or loss_value_vl <= vlss_mn:
        if acc_vl >= vacc_mx and loss_value_vl <= vlss_mn:
            vacc_early = acc_vl
            vlss_early = loss_value_vl
            epoch_early = epoch + 1
            saver.save(sess, checkpt_file)
        vacc_mx = np.max((acc_vl, vacc_mx))
        vlss_mn = np.min((loss_value_vl, vlss_mn))
        curr_step = 0
    else:
        curr_step += 1
        if curr_step == FLAGS.patience:
            print('Early stop!')
            break

saver.restore(sess, checkpt_file)
t_tst_s = time.time()
bbias = biases
loss_value_ts, acc_ts = sess.run([loss, accuracy],
    feed_dict={
        ftr_in: features[0:1],
        bias_in: bbias,
        adj_all_in : adj_all,
        adj_neig_in : adj_neig,
        N_all_in : N_all,
        lbl_in: y_test[0:1],
        msk_in: test_mask[0:1],
        is_train: False,
        attn_drop: 0.0, ffd_drop: 0.0})
t_tst_e = time.time()
print("test_loss=", "{:.5f}".format(loss_value_ts), "test_acc=", "{:.4f}".format(acc_ts))
sess.close()
t_rep_e = time.time()
print('early stop #epoch', epoch_early, 'val_loss=', "{:.5f}".format(vlss_early), 'val_acc=',"{:.4f}".format(vacc_early))
