import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import tensorflow as tf

#加载minst数据
mnist = tf.keras.datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()


#初始
plt.rcParams['font.sans-serif'] = ["SimHei"]

plt.figure(num = "xgq")

for i in range(16):
    num = np.random.randint(1, 10000)

    plt.subplot(4, 4, i+1)
    plt.axis("off")
    plt.imshow(test_x[num], cmap="gray")
    plt.title(("标签值:{}".format(test_y[num])), fontsize=14)



plt.tight_layout(rect = [0, 0, 1, 0.9])
plt.suptitle("MNIST测试集样本", fontsize=20, color='red')
plt.show()

# print("train_y:", train_x.shape, train_x.dtype)
# print("train_y:", train_y.shape, train_y.dtype)

# print(train_x.shape)
# print(test_x.shape)