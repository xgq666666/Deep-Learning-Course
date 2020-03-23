import numpy as np
import tensorflow as tf

#判断输入规则
def inputM():
    
    try:
        x = float(input("请输入商品房面积（20-500之间的实数）："))
        if x<20 or x>500:
            return 1
        return x
    except:
        return 0

def inputF():
    try:
        x = int(input("请输入商品房房间数（1-10之间的整数）"))
        if x<1 or x>10:
            return -1
        return x
    except:
        return 0


#数据
x1_np = np.array([137.97, 104.50, 100.00, 124.32, 79.20, 99.00, 124.00, 114.00, 106.69, 138.05, 53.75, 46.91, 68.00, 63.02, 81.26, 86.21])
x2_np = np.array([3, 2, 2, 3, 1, 2, 3, 2, 2, 3, 1, 1, 1, 1, 2, 2])
x0_np = np.ones(len(x1_np))
y_np = np.array([145.00, 110.00, 93.00, 116.00, 65.32, 104.00, 118.00, 91.00, 62.00, 133.00, 51.00, 45.00, 78.50, 69.65, 75.69, 95.30])

X_np = np.stack((x0_np, x1_np, x2_np), axis=1)
Y_np = y_np.reshape(-1, 1)

#转换成张量进行运算
X = tf.constant(value=X_np, dtype=tf.float32)
Y = tf.constant(value=Y_np, dtype=tf.float32)

W = tf.linalg.inv(tf.transpose(X) @ X) @ tf.transpose(X) @ Y


W_np = W.numpy().reshape(-1)

print('-'*15, 'start', '-'*15)
#判断用户输入的内容提示相关信息
x1  = inputM()
while x1==0 or x1==1:
    if x1==0:
        print("您输入的类型错误。。。OvO")
    else:
        print("您输入的范围不正确。。。OwO")
    x1 = inputM()

x2 = inputF()
while x2==0 or x2==-1:
    if x2==0:
        print("您输入的类型错误。。。OvO")
    else:
        print("您输入的范围不正确。。。OwO")
    x2 = inputF()


#进行预测值计算输出
y = W_np[1]*x1 + W_np[2]*x2 + W_np[0]

print("面积<{}>房间数<{}>--预测价格：".format(x1, x2), round(y, 2), "万元")

print('-'*15, 'end', '-'*15)

# print(x1)
# print(x2)


# print(W)
# print(len(x2_np))
# print(Y_np.shape)

