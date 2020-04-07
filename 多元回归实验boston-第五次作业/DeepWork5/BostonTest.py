import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import scale

#加载波士顿房价数据
boston_housing = tf.keras.datasets.boston_housing
(x_data, y_data), (_, _) = boston_housing.load_data(test_split=0)

# 划分数据
train_num = 300
valid_num = 100
test_num = len(x_data) - train_num - valid_num

# 训练数据集
x_train = x_data[:train_num]
y_train = y_data[:train_num]

# 验证数据集
x_valid = x_data[train_num:train_num+valid_num]
y_valid = y_data[train_num:train_num+valid_num]

# 测试数据集
x_test = x_data[train_num+valid_num:train_num+valid_num+test_num]
y_test = y_data[train_num+valid_num:train_num+valid_num+test_num]

# 转换类型并进行数据处理（归一化）
x_train = tf.cast(scale(x_train), dtype=tf.float32)
x_valid = tf.cast(scale(x_valid), dtype=tf.float32)
x_test = tf.cast(scale(x_test), dtype=tf.float32)

# 构造模型
def model(x, w, b):
    return tf.matmul(x, w) + b

# 优化变量
W = tf.Variable(tf.random.normal([13, 1], mean=0.0, stddev=1.0, dtype=tf.float32))
B = tf.Variable(tf.zeros(1), dtype=tf.float32)

# 超参数
training_epochs = 100    # 迭代次数
learning_rate = 0.01    # 学习率
batch_size = 30    # 每批次样本数

# 损失函数
def loss(x, y, w, b):
    err = model(x, w, b) - y
    squared_err = tf.square(err)
    return tf.reduce_mean(squared_err)

# 梯度
def grad(x, y, w, b):
    with tf.GradientTape() as tape:
        loss_ = loss(x, y, w, b)
    return tape.gradient(loss_, [w, b])

# 优化器
optimizer = tf.keras.optimizers.SGD(learning_rate)

# loss保存
loss_list_train = []
loss_list_valid = []
total_step = int(train_num/batch_size)

for epoch in range(training_epochs):
    for step in range(total_step):
        xs = x_train[step*batch_size:(step+1)*batch_size, :]
        ys = y_train[step*batch_size:(step+1)*batch_size]
        
        grads = grad(xs, ys, W, B)    # 计算梯度
        optimizer.apply_gradients(zip(grads, [W, B]))  # 调整（下降)
    loss_train = loss(x_train, y_train, W, B).numpy()
    loss_valid = loss(x_valid, y_valid, W, B).numpy()
    loss_list_train.append(loss_train)
    loss_list_valid.append(loss_valid)
    print("epoch={:3d}, train_loss={:.4f}, valid_loss={:.4f}".format(epoch+1, loss_train, loss_valid))

# 测试数据集损失
print("\nTest Loss", loss(x_test, y_test, W, B).numpy())

test_house_id = np.random.randint(0, test_num)
y = y_test[test_house_id]
y_pred = model(x_test, W, B)[test_house_id]
y_predit = tf.reshape(y_pred, ()).numpy()
print("\nHouse id", test_house_id, "Actual value", y, "Predicted value", y_predit)

# 损失变化
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(loss_list_train, 'blue', label="Train Loss")
plt.plot(loss_list_valid, 'red', label="Valid Loss")
plt.show()