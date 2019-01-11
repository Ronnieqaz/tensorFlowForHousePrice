import tensorflow as tf
import numpy as np
import xlrd
import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#x_data = np.random.randn(10, 3)
x_data = np.array([[4,136,22],[2,137,27],[5,127,18]])
model_folder = './weights'
checkpoint = tf.train.get_checkpoint_state(model_folder)
input_checkpoint = checkpoint.model_checkpoint_path

saver = tf.train.import_meta_graph(
    input_checkpoint + '.meta', clear_devices=True)
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
saver.restore(sess, input_checkpoint)

graph = tf.get_default_graph()
prob_tensor = graph.get_tensor_by_name("output:0")
inputs = graph.get_tensor_by_name("input:0")
pred = sess.run(prob_tensor,feed_dict={inputs:x_data})
print(pred)
