import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

#加载波士顿房价数据
boston_housing = tf.keras.datasets.boston_housing
(x, y), (_, _) = boston_housing.load_data(test_split=0)

#初始属性
plt.rcParams['font.sans-serif'] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False

titles = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "TAX", "PTRATIO", "B-1000", "LSTAT", "MEDV"]

#------------------------------------------------------------1
#显示各种属性与房价关系
plt.figure(figsize=(12, 12))

for i in range(13):
    plt.subplot(4, 4, (i+1))
    plt.scatter(x[:, i], y)
    plt.xlabel(titles[i])
    plt.ylabel("Price($1000's)")
    plt.title(str(i+1) + "." + titles[i] + " - Price")

plt.tight_layout(rect = [0, 0, 1, 0.9])

# plt.tight_layout()

plt.suptitle("各种属性与房价的关系", x=0.5, fontsize=20)
plt.show()


#---------------------------------------------------------------2
for i in range(13):
    print(i+1, "-", titles[i])

inr = int(input("请选择属性："))

#显示指定属性与房价关系
plt.figure(num = titles[inr-1])
plt.subplot(111)
plt.scatter(x[:, inr-1], y)
plt.xlabel(titles[inr-1])
plt.ylabel("Price($1000's)")
plt.title(str(inr) + "." + titles[inr-1] + " - Price")
plt.show()






# print(type(train_x))
# print("Training set", len(train_x))
# print(train_x[1])
# print("Testing set", len(test_x))