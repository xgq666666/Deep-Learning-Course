import numpy as np
import tensorflow as tf


x_np = np.array([64.3, 99.6, 145.45, 63.75, 135.46, 92.85, 86.97, 144.76, 59.3, 116.03])

y_np = np.array([62.55, 82.42, 132.62, 73.31, 131.05, 86.57, 85.49, 127.44, 55.25, 104.84])
#利用numpy数组创建张量并进行运算
x = tf.constant(value=x_np, dtype=tf.float32)
y = tf.constant(value=y_np, dtype=tf.float32)


#x,y的均值
xm = tf.reduce_mean(x)
ym = tf.reduce_mean(y)

#计算w
s1 = tf.reduce_sum(((x - xm) * (y - ym)))
s2 = tf.reduce_sum((x - xm) ** 2)
w = s1 / s2

#计算b
b = ym - (w * xm)

print("\n", "-"*13, "start", "-"*13, "\n")
print("W = ", w.numpy())
print("b = ", b.numpy())
print("\n", "-"*14, "end", "-"*14, "\n")