import argparse
import numpy as np
import os
import random


def get_params():
    ''' Get parameters from command line '''
    parser = argparse.ArgumentParser()

    # settings
    parser.add_argument("--dataset", type=str, default='cora', help="Dataset string")# 'cora', 'citeseer', 'pubmed'
    parser.add_argument('--id', type=str, default='default_id', help='id to store in database')  #
    parser.add_argument('--device', type=int, default=-1,help='device to use')  #
    parser.add_argument('--setting', type=str, default="description of hyper-parameters.")  #
    parser.add_argument('--task_type', type=str, default='semi')
    parser.add_argument('--early_stop', type=int, default= 200, help='early_stop')
    parser.add_argument('--dtype', type=str, default='float32')  #
    parser.add_argument('--seed',type=int, default=1234, help='seed')
    parser.add_argument('--record',type=bool, default=False, help='write to database, for tuning.')

    parser.add_argument('--emb', type=int, default=512, help='embdding dimention') #512

    # shared parameters
    parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
    parser.add_argument('--dropout',type=float, default=0., help='dropout rate (1 - keep probability).')
    parser.add_argument('--weight_decay',type=float, default=0.00, help='Weight for L2 loss on embedding matrix.')#0.0
    parser.add_argument('--hiddens', type=str, default='256')#256
    # 0.01 0.05 0.1 0.02
    parser.add_argument("--lr", type=float, default=0.01,help='initial learning rate.')#0.01
    parser.add_argument('--act', type=str, default='relu', help='activation funciton')  #
    parser.add_argument('--initializer', default='glorot')


    # for dropedge
    parser.add_argument('--dropedge',type=float, default=0.1, help='dropedge rate (1 - keep probability).')


    # for PTDNet
    parser.add_argument('--init_temperature', type=float, default=0.67)
    parser.add_argument('--temperature_decay', type=float, default=1.0)#1.0
    parser.add_argument('--denoise_hidden_1', type=int, default=64) #64
    parser.add_argument('--denoise_hidden_2', type=int, default=0)
    # parser.add_argument('--denoise_bias', type=float, default=1.0,help='initial ratio of edges')

    parser.add_argument('--gamma', type=float, default=-0.2) #-0.1 -0.2
    parser.add_argument('--zeta', type=float, default=2.0) #3.0 2.0

    parser.add_argument('--lambda1', type=float, default=0.003, help='Weight for L0 loss on laplacian matrix.')#0.003
    parser.add_argument('--lambda3', type=float, default=0.00, help='Weight for nuclear loss') #0 0.002 0.00005
    parser.add_argument('--k_svd', type=int, default=2)

    args, _ = parser.parse_known_args()
    return args

args = get_params()
params = vars(args)
SVD_PI = True
devices = ['0','1','-1']

real_device = args.device%len(devices)
os.environ["CUDA_VISIBLE_DEVICES"] = devices[real_device]
import tensorflow as tf

config = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=8,
    inter_op_parallelism_threads=8)
config.gpu_options.allow_growth = True
# tf.enable_eager_execution(config=config)

seed = args.seed
random.seed(args.seed)
np.random.seed(seed)
tf.random.set_seed(seed)

dtype = tf.float32
if args.dtype=='float64':
    dtype = tf.float64

eps = 1e-7

# import mysql.connector
# mydb = mysql.connector.connect(
#   host="104.39.196.38", #"",
#   user="dul262",
#   passwd="dul262dgx1"
# )
# mycursor = mydb.cursor()
