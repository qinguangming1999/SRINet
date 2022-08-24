from config import *
from utils import *
from models import SRINetGCN
from metrics import *
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from scipy.io import loadmat

# Settings
austin_my = r'data/austin_data_my.mat'
nyc_my = r'data/nyc_data_my.mat'
sf_la_my = r'data/sf_la_data_my.mat'
data = loadmat(austin_my)

adj = data['adj']
features = data['features']
adj_friendship = data['adj_friendship']
# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()
adj_train, train_edges = sample_edges(adj)
adj = adj_train
adj_train_friendship, train_edges_friendship, train_edges_friendship_false, val_edges, val_edges_false, test_edges, test_edges_false = \
    sample_friendship_edges(adj_friendship)
adj_friendship = adj_train_friendship

adj_label = adj_train + sp.eye(adj_train.shape[0])
labels = adj_label.todense()

adj_label_friendship = adj_train_friendship + sp.eye(adj_train_friendship.shape[0])
labels_friendship = adj_label_friendship.todense()

nodesize = features.shape[0]

# Some preprocessing
features = preprocess_features(features)
support = preprocess_graph(adj)
support_friendship = preprocess_graph(adj_friendship)

model = SRINetGCN(input_dim=features.shape[1], output_dim=args.emb)
optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
tuple_adj = sparse_to_tuple(adj.tocoo())
tuple_adj_friendship = sparse_to_tuple(adj_friendship.tocoo())
features_tensor = tf.convert_to_tensor(features, dtype=dtype)
adj_tensor = tf.SparseTensor(*tuple_adj)
adj_tensor_friendship = tf.SparseTensor(*tuple_adj_friendship)
label_tensor = tf.convert_to_tensor(labels, dtype=tf.float32)
label_tensor_friendship = tf.convert_to_tensor(labels_friendship, dtype=tf.float32)
flatten_label_tensor = tf.reshape(label_tensor, [-1])
flatten_label_tensor_friendship = tf.reshape(label_tensor_friendship, [-1])

model.set_fea_adj(np.array(range(adj.shape[0])), features_tensor, adj_tensor)
model.set_high_adj_values(sampleHighAdj(adj_tensor))
model.set_fea_adj_f(np.array(range(adj_friendship.shape[0])), adj_tensor_friendship)
pos_weight_f = float(adj_friendship.shape[0] * adj_friendship.shape[0] - adj_friendship.sum()) / adj_friendship.sum()
norm_f = adj_friendship.shape[0] * adj_friendship.shape[0] / float((adj_friendship.shape[0] * adj_friendship.shape[0] -
                                                                    adj_friendship.sum()) * 2)


def get_roc_score(edges_pos, edges_neg, emb):
    emb = emb.numpy()

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
    preds_neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    return roc_score, ap_score


best_val_roc_curr = 0
curr_step = 0
for epoch in range(args.epochs):
    temperature = args.init_temperature * pow(args.temperature_decay, epoch)

    with tf.GradientTape() as tape:
        emb, pred = model.call(temperature, training=True)

        cross_loss_f = norm_f * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits=pred, labels=flatten_label_tensor_friendship
                                                     , pos_weight=pos_weight_f))
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables])
        lossl0 = model.lossl0(temperature)
        nuclear = model.nuclear()

        loss = cross_loss_f + args.weight_decay * lossL2 + args.lambda1 * lossl0 + args.lambda3 * nuclear
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    emb, pred = model.call(None, training=False)

    edges_volumn = tf.reduce_sum(model.maskes[0])
    print('edge_vol', edges_volumn.numpy())

    correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(pred), 0.5), tf.int32),
                                  tf.cast(flatten_label_tensor_friendship, tf.int32))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    val_roc_curr, val_ap_curr = get_roc_score(val_edges, val_edges_false, emb)
    roc_curr, ap_curr = get_roc_score(test_edges, test_edges_false, emb)
    if val_roc_curr > best_val_roc_curr:
        curr_step = 0
        best_val_roc_curr = val_roc_curr
        best_test_roc_curr, best_ap_curr = get_roc_score(test_edges, test_edges_false, emb)

    else:
        curr_step += 1
    if curr_step > args.early_stop:
        print("Early stopping...")
        break

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(float(loss)),
          "val_roc_curr=", "{:.5f}".format(val_roc_curr),
          "val_ap_curr=", "{:.5f}".format(val_ap_curr), "best_test_roc_curr=", "{:.5f}".format(best_test_roc_curr),
          "best_ap_curr=", "{:.5f}".format(best_ap_curr))