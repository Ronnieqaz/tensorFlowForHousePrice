[TOC]

-----
# 1. 问题的提出
假设你是一名房地产经纪人，你的生意蒸蒸日上，因此你雇了一批新员工来帮忙。但是问题来了——虽然你可以一眼估算出房子的价格，
但新员工却不像你这样经验丰富，他们不知道如何给房子估价。

| 卧室数量 | 房屋面积 | 阳台面积 | 成交价 |
| --- | --- | --- | --- |
| 4 | 150 | 30 | ？？？ |

近三个月来，每当你的城市里有人卖了房子，你都记录了下面的细节——卧室数量、房屋面积、阳台面积等等。但最重要的是，你写下
了最终的成交价：

|卧室数量  |房屋面积  |阳台面积  | 成交价 |
| --- | --- | --- | --- |
| 3 | 130| 20 | 100w |
| 2 | 120 | 15 | 90w |
| 1 | 108 | 13 | 85w |

为了帮助你的新员工（也许就是为了给自己放个假），你决定使用TensorFlow写一个可以根据房屋卧室数量、房屋面积以及阳台
面积等因素来评估一间房屋的价格的ML程序。
# 2. TensorFlow解决思路
这个问题要解决的就是找到成交价（Prize）和卧室数量（RoomNum）、房屋面积（AreaHouse）、阳台面积（AreaBalcony）
之间的关系。就像是小学时做的数学测验：
![6c0309b17764fcd1d415b3989e39e58a.jpeg](evernotecid://0F7C84DB-9663-4FC7-B768-B8A6DD0BD29F/appyinxiangcom/20498392/ENNote/p145?hash=6c0309b17764fcd1d415b3989e39e58a)
然后给出任意的卧室数量、房屋面积、阳台面积，就能确定成交价了。

使用TensorFlow解决此问题的步骤包括：
* 准备数据（包括训练数据和测试数据）
* 规划神经网络
* 利用TensorFlow框架建模分析
* 测试
## 2.1 准备数据
本次的训练数据（测试数据）是一张excel表，如下图所示：
![1b21b7daad58461e7c007acc94253919.png](evernotecid://0F7C84DB-9663-4FC7-B768-B8A6DD0BD29F/appyinxiangcom/20498392/ENResource/p436)
## 2.2 规划神经网络
### 2.2.1 神经网络架构
* 输入层：
有卧室数量、房屋面积、阳台面积三个参数，所以，输入层3个神经元；
* 隐含层：
为了简单起见，1个隐层，隐层包括20个神经元（下图中由于篇幅，隐层只列出10个神经元）；
* 输出层
只有成交价1个输出，所以，1个神经元。
![799d7d7aa0b1dea97ea6d463ecb5b8df.png](evernotecid://0F7C84DB-9663-4FC7-B768-B8A6DD0BD29F/appyinxiangcom/20498392/ENResource/p437)
### 2.2.2 激活函数与代价计算
* 激活函数
选用relu（）函数，relu是广泛使用的激活函数，关于relu函数请参考机器学习的相关介绍；
* 代价计算
选用均方误差作为成本，以y表示成交价，y'表示预测成交价，则：
```math
cost = E(y - y')^{2}
```
## 2.3 利用TensorFlow框架建模
### 2.3.1 导入tensorflow以及几个常用python包
```python
#-*- coding:utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import xlrd
import argparse
import os
```
### 2.3.2 编写读取Excel表数据函数
```python
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
    return X_train, y_trai
```
### 2.3.2 编写建立神经网络函数
```python
def build_net():
    w1 = tf.Variable(tf.random_normal([3, 20]), name='w1')
    b1 = tf.Variable(tf.random_normal([20]), name='b1')
    w2 = tf.Variable(tf.random_normal([20, 1]), name='w2')
    b2 = tf.Variable(tf.random_normal([1]), name='b2')
    
    hid1 = tf.add(tf.matmul(x, w1), b1)
    hid1_out = tf.nn.relu(hid1) 
    y_ = tf.add(tf.matmul(hid1_out, w2), b2,name='output')

    return y_

```
### 2.3.4 设置运行参数
```python
# 设置运行参数
parser = argparse.ArgumentParser()
# 学习率
parser.add_argument('--lr',
                    type=float, default=1e-3,
                    help='learning rate')
# 训练次数
parser.add_argument('--epoches',
                    type=int, default=50,
                    help='train epoches')
# 每次训练的样本数量
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

```

### 2.3.5 获取数据并利用tensorflow加工
```python
X_train, y_train = parse_excel(args.excel_path)

dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset = dataset.batch(args.batch_size)
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.repeat(args.epoches)
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
```
### 2.3.6 建立Tensorflow数据流图
```python
x = tf.placeholder(tf.float32, [None, 3],name='input')
y = tf.placeholder(tf.float32, [None, 1],name='lable')

os.environ['CUDA_VISIBLE_DEVICES']='0'#用0号GPU

predict_y = build_net()
loss_ = tf.losses.mean_squared_error(y, predict_y)
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=args.lr).minimize(loss_)

init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

saver = tf.train.Saver()
```
### 2.3.7 运行tensorflow并保存训练模型
```python
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
```


## 2.4 测试
### 2.4.1 导入必要包
```python
#-*- coding:utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import 

import tensorflow as tf
import numpy as np
import os

```
### 2.4.2 输入测试数据以及载入训练好的模型
```python
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
x_data = np.array([[4,136,22],[2,137,27],[5,127,18]])
model_folder = './weights'
checkpoint = tf.train.get_checkpoint_state(model_folder)
input_checkpoint = checkpoint.model_checkpoint_path

saver = tf.train.import_meta_graph(
    input_checkpoint + '.meta', clear_devices=True)
```
### 2.4.3 预测计算
```python
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
saver.restore(sess, input_checkpoint)

graph = tf.get_default_graph()
prob_tensor = graph.get_tensor_by_name("output:0")
inputs = graph.get_tensor_by_name("input:0")
pred = sess.run(prob_tensor,feed_dict={inputs:x_data})
print(pred)

```
## 2.5 完整代码
包括训练数据，训练tensorflow代码，测试数据，测试tensorflow代码
* github托管地址：[https://github.com/Ronnieqaz/tensorFlowForHousePrice.git](https://github.com/Ronnieqaz/tensorFlowForHousePrice.git)

