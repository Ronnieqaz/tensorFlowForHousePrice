#-*- coding:utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import xlrd
import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'

# 读取数据
def parse_excel(excel_path):
    data = xlrd.open_workbook(excel_path)
    table = data.sheets()[0]
    rows = table.nrows
    cols = table.ncols
    X_train = []
    y_train = []
    for i in range(1, rows):
        line = table.row_values(i)
        X_train.append(line[:-1])
        y_train.append(line[-1])
    X_train = np.array(X_train).astype(np.float32)
    y_train = np.array(y_train).astype(np.float32)
    y_train = y_train[:, np.newaxis]
    return X_train, y_train

# 建立神经网络
def build_net():
    w1 = tf.Variable(tf.random_normal([3, 20]), name='w1')
    b1 = tf.Variable(tf.random_normal([20]), name='b1')
    w2 = tf.Variable(tf.random_normal([20, 1]), name='w2')
    b2 = tf.Variable(tf.random_normal([1]), name='b2')
    
    hid1 = tf.add(tf.matmul(x, w1), b1)
    hid1_out = tf.nn.relu(hid1)
    y_ = tf.add(tf.matmul(hid1_out, w2), b2,name='output')

    return y_


# 设置参数
parser = argparse.ArgumentParser()
## 学习率
parser.add_argument('--lr',
                    type=float, default=1e-3,
                    help='learning rate')
## 训练次数
parser.add_argument('--epoches',
                    type=int, default=50,
                    help='train epoches')
## 每次训练的样本数量
parser.add_argument('--batch_size',
                    type=int, default=5,
                    help='train batch size')
# 训练模型保存位置
parser.add_argument('--save_folder',
                    type=str, default='weights/',
                    help='save weight path')
# 数据源位置
parser.add_argument('--excel_path',
                    type=str, default='data.xlsx',
                    help='excel path')
args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)


# 读取并处理数据
X_train, y_train = parse_excel(args.excel_path)

dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset = dataset.batch(args.batch_size)
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.repeat(args.epoches)
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()

#建立数据流图
x = tf.placeholder(tf.float32, [None, 3],name='input')
y = tf.placeholder(tf.float32, [None, 1],name='label')

os.environ['CUDA_VISIBLE_DEVICES']='0'#用0号GPU
predict_y = build_net()
loss_ = tf.losses.mean_squared_error(y, predict_y)
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=args.lr).minimize(loss_)

init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

saver = tf.train.Saver()

# 运行tensorflow
with tf.Session(config=config) as sess:
    sess.run(init)
    tloss = 0
    itertion = 0
    try:
        while True:
            inputs_data, labels = sess.run(one_element)
            _, loss = sess.run([optimizer, loss_], feed_dict={
                               x: inputs_data, y: labels})
            tloss += loss
            if itertion % 10 == 0 and itertion != 0:
                print('itertion:{} || loss:{:.4f}'.format(
                    itertion, tloss / itertion))
            if itertion % 100 == 0 and itertion != 0:
                saver.save(sess, os.path.join(args.save_folder,
                                              'mymodel'), global_step=itertion)
            itertion += 1

    except tf.errors.OutOfRangeError:
        print("end!")


：x
:x
:q

