from inits import *
import tensorflow as tf
from config import args,dtype
from tensorflow.python.keras import layers

class InnerProductDecoder(layers.Layer):
    """Decoder model layer for link prediction."""
    def __init__(self, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.act = act
        self.dropout = dropout
    def call(self, inputs):
        #inputs = tf.nn.dropout(inputs, 1-self.dropout)
        inputs = tf.nn.dropout(inputs, self.dropout)
        x = tf.transpose(inputs)
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])
        outputs = self.act(x)
        return outputs


class InnerProductDecoderFriend(layers.Layer):
    """Decoder model layer for link prediction."""
    def __init__(self, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoderFriend, self).__init__(**kwargs)
        self.act = act
        self.dropout = dropout

    def call(self, inputs):
        #inputs = tf.nn.dropout(inputs, 1-self.dropout)
        inputs = tf.nn.dropout(inputs, self.dropout)
        x = tf.transpose(inputs)
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])
        outputs = self.act(x)
        return outputs



class Dense(layers.Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, activation=tf.nn.relu, bias=False, **kwargs):
        super(Dense, self).__init__(**kwargs)


        self.act = activation
        self.bias = bias

        if args.initializer=='he':
            initializer = 'he_normal'#tf.initializers.glorot_normal()##
        else:
            initializer = tf.initializers.glorot_normal()##


        self.weight = self.add_weight('weight', [input_dim, output_dim],dtype=dtype,  initializer=initializer)
        if self.bias:
            self.bias_weight = self.add_weight('bias', [output_dim],dtype=dtype,  initializer="zero")


    def call(self, inputs, training=None):
        x = inputs

        if training and args.dropout>0:
            x = tf.nn.dropout(x, 1-args.dropout)

        # convolve
        output = tf.matmul(x, self.weight)

        if self.bias:
            output += self.bias_weight

        return self.act(output)


class GraphConvolution(layers.Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, activation=tf.nn.relu, bias=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)


        self.act = activation
        self.bias = bias

        if args.initializer=='he':
            initializer = 'he_normal'#tf.initializers.glorot_normal()##
        else:
            initializer = tf.initializers.glorot_normal()##


        self.weight = self.add_weight('weight', [input_dim, output_dim],dtype=dtype,  initializer=initializer)
        if self.bias:
            self.bias_weight = self.add_weight('bias', [output_dim],dtype=dtype,  initializer="zero")


    def call(self, inputs, training=None):
        x, support = inputs

        if training and args.dropout>0:
            x = tf.nn.dropout(x, 1-args.dropout)

        # convolve
        pre_sup = tf.matmul(x, self.weight)

        if  isinstance(support,tf.Tensor):
            output = tf.matmul(support, pre_sup)
        else:
            output = tf.sparse.sparse_dense_matmul(support, pre_sup)

        if self.bias:
            output += self.bias_weight

        return self.act(output)
