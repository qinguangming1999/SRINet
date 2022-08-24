from config import *
from layers import *
from metrics import *
from utils import samplePred
from tensorflow import keras
from scipy.sparse.linalg import svds,eigsh
from scipy.sparse import csc_matrix

class GCNAE(keras.Model):
    def __init__(self, input_dim, output_dim,**kwargs):
        super(GCNAE, self).__init__(**kwargs)

        try:
            hiddens = [int(s) for s in args.hiddens.split('-')]
        except:
            hiddens =[args.hidden1]

        self.layers_ = []

        layer0 = GraphConvolution(input_dim=input_dim,
                                  output_dim=hiddens[0],
                                  activation=tf.nn.relu)
        self.layers_.append(layer0)
        nhiddens = len(hiddens)
        for _ in range(1,nhiddens):
            layertemp = GraphConvolution(input_dim=hiddens[_-1],
                                      output_dim=hiddens[_],
                                      activation=tf.nn.relu)
            self.layers_.append(layertemp)

        layer_1 = GraphConvolution(input_dim=hiddens[-1],
                                            output_dim=output_dim,
                                            activation=lambda x: x)

        self.reconstructions = InnerProductDecoder(act=lambda x: x)

        self.layers_.append(layer_1)
        self.hiddens = hiddens

    def call(self,inputs,training=None):
        x, support  = inputs
        for layer in self.layers_:
            x = layer.call((x,support),training)
        self.embeddings = x
        pred = self.reconstructions.call(x)
        return x,pred

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])





class SRINetGCN(GCNAE):
    def __init__(self, input_dim, output_dim,activation=tf.nn.relu, **kwargs):
        super(SRINetGCN, self).__init__(input_dim,output_dim,**kwargs)

        hidden_1 = args.denoise_hidden_1
        hidden_2 = args.denoise_hidden_2

        if args.initializer=='he':
            initializer = 'he_normal'#tf.initializers.glorot_normal()##
        else:
            initializer = tf.initializers.glorot_normal()##

        self.nblayers = []
        self.selflayers = []

        self.attentions = []
        self.attentions.append([])
        for hidden in self.hiddens:
            self.attentions.append([])

        for i in range(len(self.attentions)):
            self.nblayers.append(tf.keras.layers.Dense(hidden_1, activation=activation, kernel_initializer=initializer))
            self.selflayers.append(tf.keras.layers.Dense(hidden_1, activation=activation, kernel_initializer=initializer))

            if hidden_2>0:
                self.attentions[i].append(tf.keras.layers.Dense(hidden_2, activation=activation , kernel_initializer = initializer))

            self.attentions[i].append(tf.keras.layers.Dense(1, activation=lambda x:x, kernel_initializer=initializer))

        self.attention_layers = []
        self.attention_layers.extend(self.nblayers)
        self.attention_layers.extend(self.selflayers)
        for i in range(len(self.attentions)):
            self.attention_layers.extend(self.attentions[i])


    def set_fea_adj(self,nodes,fea,adj):
        self.nodes = nodes
        self.node_size = len(nodes)
        self.features = fea
        self.adj_mat = adj
        self.row = adj.indices[:,0]
        self.col = adj.indices[:,1]


    def set_fea_adj_f(self, nodes_f, adj_f):
        self.nodes_f = nodes_f
        self.node_size_f = len(nodes_f)
        self.adj_mat_f = adj_f
        self.row_f = adj_f.indices[:, 0]
        self.col_f = adj_f.indices[:, 1]

    def set_high_adj_values(self,high_adj_values):
        self.high_adj_values = high_adj_values

    def get_attention(self, input1, input2, layer=0, training=False):

        nb_layer = self.nblayers[layer]
        selflayer = self.selflayers[layer]
        nn = self.attentions[layer]

        if tf.__version__.startswith('2.'):
            dp = args.dropout
        else:
            dp = 1 - args.dropout

        input1 = nb_layer(input1)
        if training:
            input1 = tf.nn.dropout(input1, dp)
        input2 = selflayer(input2)
        if training:
            input2 = tf.nn.dropout(input2, dp)

        input10 = tf.concat([input1, input2], axis=1)
        input = [input10]
        for layer in nn:
            input.append(layer(input[-1]))
            if training:
                input[-1] = tf.nn.dropout(input[-1], dp)
        weight10 = input[-1]
        return weight10

    def get_edges(self, input1, input2, layer=1, use_bias=True):
        weight = self.get_attention(input1, input2, layer, use_bias, training=False)
        edges = self.hard_concrete_sample(weight, training=False)
        return edges

    def hard_concrete_sample(self, log_alpha, beta=1.0, training=True):
        """Uniform random numbers for the concrete distribution"""
        gamma = args.gamma
        zeta = args.zeta

        if training:
            debug_var = eps
            bias = 0.0
            random_noise = bias+tf.random.uniform(tf.shape(log_alpha), minval=debug_var, maxval=1.0 - debug_var, dtype=dtype)
            gate_inputs = tf.math.log(random_noise) - tf.math.log(1.0 - random_noise)
            gate_inputs = (gate_inputs + log_alpha) / beta
            gate_inputs = tf.sigmoid(gate_inputs)
        else:
            gate_inputs = tf.sigmoid(log_alpha)

        stretched_values = gate_inputs * (zeta - gamma) + gamma
        cliped = tf.clip_by_value(
            stretched_values,
            clip_value_max=1.0,
            clip_value_min=0.0)
        return cliped

    def l0_norm(self, log_alpha, beta):
        gamma = args.gamma
        zeta = args.zeta
        reg_per_weight = tf.sigmoid(log_alpha - beta * tf.cast(tf.math.log(-gamma / zeta), dtype))
        return tf.reduce_mean(reg_per_weight)

    def call(self,inputs, training=None):

        if training:
            temperature = inputs
        else:
            temperature = 1.0

        self.edge_maskes = []

        self.maskes = []
        self.edge_weights = []

        x = self.features
        layer_index = 0
        for layer in self.layers_:
            f1_features = tf.gather(x, self.row)
            f2_features = tf.gather(x, self.col)
            weight = self.get_attention(f1_features, f2_features, layer=layer_index, training=training)
            mask = self.hard_concrete_sample(weight, temperature, training)
            mask_sum = tf.reduce_sum(mask)
            self.edge_weights.append(weight)
            self.maskes.append(mask)
            mask = tf.squeeze(mask)
            adj = tf.SparseTensor(indices=self.adj_mat.indices,
                                  values=mask,
                                  dense_shape=self.adj_mat.shape)
            # norm
            adj = tf.sparse.add(adj,tf.sparse.eye(self.node_size,dtype=dtype))

            row = adj.indices[:, 0]
            col = adj.indices[:, 1]

            rowsum = tf.sparse.reduce_sum(adj, axis=-1)#+1e-6
            d_inv_sqrt = tf.reshape(tf.pow(rowsum, -0.5),[-1])
            d_inv_sqrt = tf.clip_by_value(d_inv_sqrt, 0, 10.0)
            row_inv_sqrt = tf.gather(d_inv_sqrt,row)
            col_inv_sqrt = tf.gather(d_inv_sqrt,col)
            values = tf.multiply(adj.values,row_inv_sqrt)
            values = tf.multiply(values,col_inv_sqrt)

            support = tf.SparseTensor(indices=adj.indices,
                                  values=values,
                                  dense_shape=adj.shape)
            #print(support)
            x = layer.call((x,support),training)
            layer_index +=1
        self.embeddings = x
        pred = self.reconstructions.call(x)
        #pred_f = pred[:3712]
        return x,pred

    def lossl0(self,temperature):
        l0_loss = tf.zeros([], dtype=dtype)
        for weight in self.edge_weights:
            l0_loss += self.l0_norm(weight, temperature)

        return l0_loss

    def nuclear(self):

        nuclear_loss = tf.zeros([],dtype=dtype)
        values = []
        if args.lambda3==0:
            return 0
        for mask in self.maskes:
            mask = tf.squeeze(mask)
            support = tf.SparseTensor(indices=self.adj_mat.indices, values=mask,
                                      dense_shape=self.adj_mat.dense_shape)
            #support_dense = tf.compat.v1.sparse.to_dense(support)
            #support_order = tf.sparse.reorder(support)
            support_dense = tf.sparse.to_dense(support, validate_indices=False)
            support_trans = tf.transpose(support_dense)

            AA = tf.matmul(support_trans, support_dense)
            SVD_PI = False
            if SVD_PI:
                row_ind = self.adj_mat.indices[:, 0]
                col_ind = self.adj_mat.indices[:, 1]
                support_csc = csc_matrix((mask, (row_ind, col_ind)))
                k = args.k_svd
                u, s, vh = svds(support_csc, k=k)

                u = tf.stop_gradient(u)
                s = tf.stop_gradient(s)
                vh = tf.stop_gradient(vh)

                for i in range(k):
                    vi = tf.expand_dims(tf.gather(vh, i), -1)
                    for ite in range(1):
                        vi = tf.matmul(AA, vi)
                        vi_norm = tf.linalg.norm(vi)
                        vi = vi / vi_norm

                    vmv = tf.matmul(tf.transpose(vi), tf.matmul(AA, vi))
                    vv = tf.matmul(tf.transpose(vi), vi)

                    t_vi = tf.math.sqrt(tf.abs(vmv / vv))
                    values.append(t_vi)

                    if k > 1:
                        AA_minus = tf.matmul(AA, tf.matmul(vi, tf.transpose(vi)))
                        AA = AA - AA_minus
            else:
                trace = tf.linalg.trace(AA)
                values.append(tf.reduce_sum(trace))

            nuclear_loss = tf.add_n(values)

        return nuclear_loss


class GCNAEMul(keras.Model):
    def __init__(self, input_dim, output_dim, num_mul, **kwargs):
        super(GCNAEMul, self).__init__(**kwargs)

        try:
            hiddens = [int(s) for s in args.hiddens.split('-')]
        except:
            hiddens =[args.hidden1]
        self.num_mul = num_mul
        self.layers_mul = []
        for i in range(self.num_mul):
            layers_ = []
            layer0 = GraphConvolution(input_dim=input_dim,
                                      output_dim=hiddens[0],
                                      activation=tf.nn.relu)
            layers_.append(layer0)
            nhiddens = len(hiddens)
            for _ in range(1,nhiddens):
                layertemp = GraphConvolution(input_dim=hiddens[_-1],
                                          output_dim=hiddens[_],
                                          activation=tf.nn.relu)
                self.layers_.append(layertemp)

            layer_1 = GraphConvolution(input_dim=hiddens[-1],
                                                output_dim=output_dim,
                                                activation=lambda x: x)
            layers_.append(layer_1)
            self.layers_mul.append(layers_)
        self.hiddens = hiddens
        self.reconstructions = InnerProductDecoder(act=lambda x: x)

    def call(self,inputs,training=None):
        x, support  = inputs
        for layer in self.layers_:
            x = layer.call((x,support),training)
        self.embeddings = x
        pred = self.reconstructions.call(x)
        return x,pred

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])


class PTDNetGCNAEFriendMultiplex(GCNAEMul):
    def __init__(self, input_dim, output_dim, num_mul, activation=tf.nn.relu, **kwargs):
        super(PTDNetGCNAEFriendMultiplex, self).__init__(input_dim,output_dim,num_mul,**kwargs)

        hidden_1 = args.denoise_hidden_1
        hidden_2 = args.denoise_hidden_2
        #emb_w = tf.Variable()
        if args.initializer=='he':
            initializer = 'he_normal'#tf.initializers.glorot_normal()##
        else:
            initializer = tf.initializers.glorot_normal()##

        self.nblayers_mul = []
        self.selflayers_mul = []
        self.attentions_mul = []
        self.attention_layers_mul = []
        for k in range(self.num_mul):
            nblayers = []
            selflayers = []

            attentions = []
            attentions.append([])
            for hidden in self.hiddens:
                attentions.append([])

            for i in range(len(attentions)):
                nblayers.append(tf.keras.layers.Dense(hidden_1, activation=activation, kernel_initializer=initializer))
                selflayers.append(tf.keras.layers.Dense(hidden_1, activation=activation, kernel_initializer=initializer))

                if hidden_2>0:
                    attentions[i].append(tf.keras.layers.Dense(hidden_2, activation=activation , kernel_initializer = initializer))

                attentions[i].append(tf.keras.layers.Dense(1, activation=lambda x:x, kernel_initializer=initializer))

            attention_layers = []
            attention_layers.extend(nblayers)
            attention_layers.extend(selflayers)
            for i in range(len(attentions)):
                attention_layers.extend(attentions[i])
            self.nblayers_mul.append(nblayers)
            self.selflayers_mul.append(selflayers)
            self.attentions_mul.append(attentions)
            self.attention_layers_mul.append(attention_layers)

    def set_fea_adj(self,nodes,fea,adj):
        self.nodes = nodes
        self.node_size = len(nodes)
        self.features = fea
        self.adj_mat = adj
        self.row = adj.indices[:,0]
        self.col = adj.indices[:,1]

    def set_fea_adj_f(self, nodes_f, adj_f):
        self.nodes_f = nodes_f
        self.node_size_f = len(nodes_f)
        self.adj_mat_f = adj_f
        self.row_f = adj_f.indices[:, 0]
        self.col_f = adj_f.indices[:, 1]


    def set_fea_adj_mul(self,nodes,fea,adj_mul):
        self.nodes = nodes
        self.node_size = len(nodes)
        self.features = fea
        self.adj_mat_mul = adj_mul
        self.rows = []
        self.cols = []
        for adj_mat in adj_mul:
            self.row = adj_mat.indices[:,0]
            self.col = adj_mat.indices[:,1]
            self.rows.append(self.row)
            self.cols.append(self.col)

    def set_high_adj_values(self,high_adj_values):
        self.high_adj_values = high_adj_values

    def get_attention(self, input1, input2, layer=0, training=False):

        nb_layer = self.nblayers[layer]
        selflayer = self.selflayers[layer]
        nn = self.attentions[layer]

        if tf.__version__.startswith('2.'):
            dp = args.dropout
        else:
            dp = 1 - args.dropout

        input1 = nb_layer(input1)
        if training:
            input1 = tf.nn.dropout(input1, dp)
        input2 = selflayer(input2)
        if training:
            input2 = tf.nn.dropout(input2, dp)

        input10 = tf.concat([input1, input2], axis=1)
        input = [input10]
        for layer in nn:
            input.append(layer(input[-1]))
            if training:
                input[-1] = tf.nn.dropout(input[-1], dp)
        weight10 = input[-1]
        return weight10

    def get_attention_mul(self, input1, input2, layer=0, mul_index=0, training=False):

        nb_layer = self.nblayers_mul[mul_index][layer]
        selflayer = self.selflayers_mul[mul_index][layer]
        nn = self.attentions_mul[mul_index][layer]

        if tf.__version__.startswith('2.'):
            dp = args.dropout
        else:
            dp = 1 - args.dropout

        input1 = nb_layer(input1)
        if training:
            input1 = tf.nn.dropout(input1, dp)
        input2 = selflayer(input2)
        if training:
            input2 = tf.nn.dropout(input2, dp)

        input10 = tf.concat([input1, input2], axis=1)
        input = [input10]
        for layer in nn:
            input.append(layer(input[-1]))
            if training:
                input[-1] = tf.nn.dropout(input[-1], dp)
        weight10 = input[-1]
        return weight10


    def get_edges(self, input1, input2, layer=1, use_bias=True):
        weight = self.get_attention(input1, input2, layer, use_bias, training=False)
        edges = self.hard_concrete_sample(weight, training=False)
        return edges

    def hard_concrete_sample(self, log_alpha, beta=1.0, training=True):
        """Uniform random numbers for the concrete distribution"""
        gamma = args.gamma
        zeta = args.zeta

        if training:
            debug_var = eps
            bias = 0.0
            random_noise = bias+tf.random.uniform(tf.shape(log_alpha), minval=debug_var, maxval=1.0 - debug_var, dtype=dtype)
            gate_inputs = tf.math.log(random_noise) - tf.math.log(1.0 - random_noise)
            gate_inputs = (gate_inputs + log_alpha) / beta
            gate_inputs = tf.sigmoid(gate_inputs)
        else:
            gate_inputs = tf.sigmoid(log_alpha)

        stretched_values = gate_inputs * (zeta - gamma) + gamma
        cliped = tf.clip_by_value(
            stretched_values,
            clip_value_max=1.0,
            clip_value_min=0.0)
        return cliped

    def l0_norm(self, log_alpha, beta):
        gamma = args.gamma
        zeta = args.zeta
        reg_per_weight = tf.sigmoid(log_alpha - beta * tf.cast(tf.math.log(-gamma / zeta), dtype))
        return tf.reduce_mean(reg_per_weight)

    def call(self,inputs, training=None):

        if training:
            temperature = inputs
        else:
            temperature = 1.0
        self.emb_mul = []
        self.edge_weights = []
        self.maskes_mul = []
        for i in range(self.num_mul):
            x = self.features
            self.edge_maskes = []
            self.maskes = []
            layer_index = 0
            for layer in self.layers_mul[i]:
                f1_features = tf.gather(x, self.rows[i])
                f2_features = tf.gather(x, self.cols[i])
                weight = self.get_attention_mul(f1_features, f2_features, layer=layer_index, mul_index=i, training=training)
                mask = self.hard_concrete_sample(weight, temperature, training)
                mask_sum = tf.reduce_sum(mask)
                self.edge_weights.append(weight)
                self.maskes.append(mask)
                mask = tf.squeeze(mask)
                # print(self.high_adj_values)
                #mask_adj = mask + self.high_adj_values
                #mask_adj = mask * self.adj_mat
                adj = tf.SparseTensor(indices=self.adj_mat_mul[i].indices,
                                      values=mask,
                                      dense_shape=self.adj_mat_mul[i].shape)
                # norm
                adj = tf.sparse.add(adj,tf.sparse.eye(self.node_size,dtype=dtype))

                row = adj.indices[:, 0]
                col = adj.indices[:, 1]

                rowsum = tf.sparse.reduce_sum(adj, axis=-1)#+1e-6
                d_inv_sqrt = tf.reshape(tf.pow(rowsum, -0.5),[-1])
                d_inv_sqrt = tf.clip_by_value(d_inv_sqrt, 0, 10.0)
                row_inv_sqrt = tf.gather(d_inv_sqrt,row)
                col_inv_sqrt = tf.gather(d_inv_sqrt,col)
                values = tf.multiply(adj.values,row_inv_sqrt)
                values = tf.multiply(values,col_inv_sqrt)

                support = tf.SparseTensor(indices=adj.indices,
                                      values=values,
                                      dense_shape=adj.shape)
                #print(support)
                x = layer.call((x,support),training)
                layer_index +=1
            self.maskes_mul.append(self.maskes)
            self.emb_mul.append(x)
        emb_sum = tf.zeros((args.emb,))
        for emb in self.emb_mul:
            emb_sum = emb_sum + emb
        emb_mean = emb_sum / self.num_mul
        #self.embeddings = x
        self.embeddings = emb_mean
        #pred = self.reconstructions.call(x)
        pred = self.reconstructions.call(emb_mean)
        return emb_mean,pred

    def lossl0(self,temperature):
        l0_loss = tf.zeros([], dtype=dtype)
        for weight in self.edge_weights:
            l0_loss += self.l0_norm(weight, temperature)

        return l0_loss

    def nuclear(self):

        nuclear_loss = tf.zeros([],dtype=dtype)
        values = []
        if args.lambda3==0:
            return 0
        for mask in self.maskes:
            mask = tf.squeeze(mask)
            support = tf.SparseTensor(indices=self.adj_mat.indices, values=mask,
                                      dense_shape=self.adj_mat.dense_shape)
            support_dense = tf.compat.v1.sparse.to_dense(support)
            support_trans = tf.transpose(support_dense)

            AA = tf.matmul(support_trans, support_dense)
            if SVD_PI:
                row_ind = self.adj_mat.indices[:, 0]
                col_ind = self.adj_mat.indices[:, 1]
                support_csc = csc_matrix((mask, (row_ind, col_ind)))
                k = args.k_svd
                u, s, vh = svds(support_csc, k=k)

                u = tf.stop_gradient(u)
                s = tf.stop_gradient(s)
                vh = tf.stop_gradient(vh)

                for i in range(k):
                    vi = tf.expand_dims(tf.gather(vh, i), -1)
                    for ite in range(1):
                        vi = tf.matmul(AA, vi)
                        vi_norm = tf.linalg.norm(vi)
                        vi = vi / vi_norm

                    vmv = tf.matmul(tf.transpose(vi), tf.matmul(AA, vi))
                    vv = tf.matmul(tf.transpose(vi), vi)

                    t_vi = tf.math.sqrt(tf.abs(vmv / vv))
                    values.append(t_vi)

                    if k > 1:
                        AA_minus = tf.matmul(AA, tf.matmul(vi, tf.transpose(vi)))
                        AA = AA - AA_minus
            else:
                trace = tf.linalg.trace(AA)
                values.append(tf.reduce_sum(trace))

            nuclear_loss = tf.add_n(values)

        return nuclear_loss

class SRINetMultiplex(GCNAEMul):
    def __init__(self, input_dim, output_dim, num_mul, activation=tf.nn.relu, **kwargs):
        super(SRINetMultiplex, self).__init__(input_dim,output_dim,num_mul,**kwargs)

        hidden_1 = args.denoise_hidden_1
        hidden_2 = args.denoise_hidden_2

        if args.initializer=='he':
            initializer = 'he_normal'#tf.initializers.glorot_normal()##
        else:
            initializer = tf.initializers.glorot_normal()##

        self.nblayers_mul = []
        self.selflayers_mul = []
        self.attentions_mul = []
        self.attention_layers_mul = []
        for k in range(self.num_mul):
            nblayers = []
            selflayers = []

            attentions = []
            attentions.append([])
            for hidden in self.hiddens:
                attentions.append([])

            for i in range(len(attentions)):
                nblayers.append(tf.keras.layers.Dense(hidden_1, activation=activation, kernel_initializer=initializer))
                selflayers.append(tf.keras.layers.Dense(hidden_1, activation=activation, kernel_initializer=initializer))

                if hidden_2>0:
                    attentions[i].append(tf.keras.layers.Dense(hidden_2, activation=activation , kernel_initializer = initializer))

                attentions[i].append(tf.keras.layers.Dense(1, activation=lambda x:x, kernel_initializer=initializer))

            attention_layers = []
            attention_layers.extend(nblayers)
            attention_layers.extend(selflayers)
            for i in range(len(attentions)):
                attention_layers.extend(attentions[i])
            self.nblayers_mul.append(nblayers)
            self.selflayers_mul.append(selflayers)
            self.attentions_mul.append(attentions)
            self.attention_layers_mul.append(attention_layers)

    def set_fea_adj(self,nodes,fea,adj):
        self.nodes = nodes
        self.node_size = len(nodes)
        self.features = fea
        self.adj_mat = adj
        self.row = adj.indices[:,0]
        self.col = adj.indices[:,1]

    def set_fea_adj_f(self, nodes_f, adj_f):
        self.nodes_f = nodes_f
        self.node_size_f = len(nodes_f)
        self.adj_mat_f = adj_f
        self.row_f = adj_f.indices[:, 0]
        self.col_f = adj_f.indices[:, 1]


    def set_fea_adj_mul(self,nodes,fea,adj_mul):
        self.nodes = nodes
        self.node_size = len(nodes)
        self.features = fea
        self.adj_mat_mul = adj_mul
        self.rows = []
        self.cols = []
        for adj_mat in adj_mul:
            self.row = adj_mat.indices[:,0]
            self.col = adj_mat.indices[:,1]
            self.rows.append(self.row)
            self.cols.append(self.col)

    def set_high_adj_values(self,high_adj_values):
        self.high_adj_values = high_adj_values

    def get_attention(self, input1, input2, layer=0, training=False):

        nb_layer = self.nblayers[layer]
        selflayer = self.selflayers[layer]
        nn = self.attentions[layer]

        if tf.__version__.startswith('2.'):
            dp = args.dropout
        else:
            dp = 1 - args.dropout

        input1 = nb_layer(input1)
        if training:
            input1 = tf.nn.dropout(input1, dp)
        input2 = selflayer(input2)
        if training:
            input2 = tf.nn.dropout(input2, dp)

        input10 = tf.concat([input1, input2], axis=1)
        input = [input10]
        for layer in nn:
            input.append(layer(input[-1]))
            if training:
                input[-1] = tf.nn.dropout(input[-1], dp)
        weight10 = input[-1]
        return weight10

    def get_attention_mul(self, input1, input2, layer=0, mul_index=0, training=False):

        nb_layer = self.nblayers_mul[mul_index][layer]
        selflayer = self.selflayers_mul[mul_index][layer]
        nn = self.attentions_mul[mul_index][layer]

        if tf.__version__.startswith('2.'):
            dp = args.dropout
        else:
            dp = 1 - args.dropout

        input1 = nb_layer(input1)
        if training:
            input1 = tf.nn.dropout(input1, dp)
        input2 = selflayer(input2)
        if training:
            input2 = tf.nn.dropout(input2, dp)

        input10 = tf.concat([input1, input2], axis=1)
        input = [input10]
        for layer in nn:
            input.append(layer(input[-1]))
            if training:
                input[-1] = tf.nn.dropout(input[-1], dp)
        weight10 = input[-1]
        return weight10


    def get_edges(self, input1, input2, layer=1, use_bias=True):
        weight = self.get_attention(input1, input2, layer, use_bias, training=False)
        edges = self.hard_concrete_sample(weight, training=False)
        return edges

    def hard_concrete_sample(self, log_alpha, beta=1.0, training=True):
        """Uniform random numbers for the concrete distribution"""
        gamma = args.gamma
        zeta = args.zeta

        if training:
            debug_var = eps
            bias = 0.0
            random_noise = bias+tf.random.uniform(tf.shape(log_alpha), minval=debug_var, maxval=1.0 - debug_var, dtype=dtype)
            gate_inputs = tf.math.log(random_noise) - tf.math.log(1.0 - random_noise)
            gate_inputs = (gate_inputs + log_alpha) / beta
            gate_inputs = tf.sigmoid(gate_inputs)
        else:
            gate_inputs = tf.sigmoid(log_alpha)

        stretched_values = gate_inputs * (zeta - gamma) + gamma
        cliped = tf.clip_by_value(
            stretched_values,
            clip_value_max=1.0,
            clip_value_min=0.0)
        return cliped

    def l0_norm(self, log_alpha, beta):
        gamma = args.gamma
        zeta = args.zeta
        reg_per_weight = tf.sigmoid(log_alpha - beta * tf.cast(tf.math.log(-gamma / zeta), dtype))
        return tf.reduce_mean(reg_per_weight)

    def call(self,inputs, training=None):

        if training:
            temperature = inputs
        else:
            temperature = 1.0
        self.emb_mul = []
        self.edge_weights = []
        self.maskes_mul = []
        for i in range(self.num_mul):
            x = self.features
            self.edge_maskes = []
            self.maskes = []
            layer_index = 0
            for layer in self.layers_mul[i]:
                f1_features = tf.gather(x, self.rows[i])
                f2_features = tf.gather(x, self.cols[i])
                weight = self.get_attention_mul(f1_features, f2_features, layer=layer_index, mul_index=i, training=training)
                mask = self.hard_concrete_sample(weight, temperature, training)
                mask_sum = tf.reduce_sum(mask)
                self.edge_weights.append(weight)
                self.maskes.append(mask)
                mask = tf.squeeze(mask)

                adj = tf.SparseTensor(indices=self.adj_mat_mul[i].indices,
                                      values=mask,
                                      dense_shape=self.adj_mat_mul[i].shape)
                # norm
                adj = tf.sparse.add(adj,tf.sparse.eye(self.node_size,dtype=dtype))

                row = adj.indices[:, 0]
                col = adj.indices[:, 1]

                rowsum = tf.sparse.reduce_sum(adj, axis=-1)#+1e-6
                d_inv_sqrt = tf.reshape(tf.pow(rowsum, -0.5),[-1])
                d_inv_sqrt = tf.clip_by_value(d_inv_sqrt, 0, 10.0)
                row_inv_sqrt = tf.gather(d_inv_sqrt,row)
                col_inv_sqrt = tf.gather(d_inv_sqrt,col)
                values = tf.multiply(adj.values,row_inv_sqrt)
                values = tf.multiply(values,col_inv_sqrt)

                support = tf.SparseTensor(indices=adj.indices,
                                      values=values,
                                      dense_shape=adj.shape)
                #print(support)
                x = layer.call((x,support),training)
                layer_index +=1
            self.maskes_mul.append(self.maskes)
            self.emb_mul.append(x)
        emb_sum = tf.zeros((args.emb,))
        for emb in self.emb_mul:
            emb_sum = emb_sum + emb
        emb_mean = emb_sum / self.num_mul
        self.embeddings = emb_mean
        pred = self.reconstructions.call(emb_mean)
        return emb_mean,pred

    def lossl0(self,temperature):
        l0_loss = tf.zeros([], dtype=dtype)
        for weight in self.edge_weights:
            l0_loss += self.l0_norm(weight, temperature)

        return l0_loss

    def nuclear(self):

        nuclear_loss = tf.zeros([],dtype=dtype)
        values = []
        if args.lambda3==0:
            return 0
        for mask in self.maskes:
            mask = tf.squeeze(mask)
            support = tf.SparseTensor(indices=self.adj_mat.indices, values=mask,
                                      dense_shape=self.adj_mat.dense_shape)
            support_dense = tf.compat.v1.sparse.to_dense(support)
            support_trans = tf.transpose(support_dense)

            AA = tf.matmul(support_trans, support_dense)
            if SVD_PI:
                row_ind = self.adj_mat.indices[:, 0]
                col_ind = self.adj_mat.indices[:, 1]
                support_csc = csc_matrix((mask, (row_ind, col_ind)))
                k = args.k_svd
                u, s, vh = svds(support_csc, k=k)

                u = tf.stop_gradient(u)
                s = tf.stop_gradient(s)
                vh = tf.stop_gradient(vh)

                for i in range(k):
                    vi = tf.expand_dims(tf.gather(vh, i), -1)
                    for ite in range(1):
                        vi = tf.matmul(AA, vi)
                        vi_norm = tf.linalg.norm(vi)
                        vi = vi / vi_norm

                    vmv = tf.matmul(tf.transpose(vi), tf.matmul(AA, vi))
                    vv = tf.matmul(tf.transpose(vi), vi)

                    t_vi = tf.math.sqrt(tf.abs(vmv / vv))
                    values.append(t_vi)

                    if k > 1:
                        AA_minus = tf.matmul(AA, tf.matmul(vi, tf.transpose(vi)))
                        AA = AA - AA_minus
            else:
                trace = tf.linalg.trace(AA)
                values.append(tf.reduce_sum(trace))

            nuclear_loss = tf.add_n(values)

        return nuclear_loss
