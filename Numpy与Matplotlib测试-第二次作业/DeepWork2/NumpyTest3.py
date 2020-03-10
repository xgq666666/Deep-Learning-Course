import numpy as np

#初始
x0 = np.ones(10)
x1 = np.array([64.3, 99.6, 145.45, 63.75, 135.46, 92.85, 86.97, 144.76, 59.3, 116.03])
x2 = np.array([2, 3, 4, 2, 3, 4, 2, 4, 1, 3])
y = np.array([62.55, 82.42, 132.62, 73.31, 131.05, 86.57, 85.49, 127.44, 55.25, 104.84])

#堆叠x0, x1, x2
X = np.stack((x0, x1, x2), axis=1)

#变换y
Y = y.reshape(10, 1)

#计算W

XM = np.mat(X)  #转换成矩阵对象
YM = np.mat(Y)

W = (XM.T * X).I * X.T * Y


print("\n", "-"*13, "start", "-"*13, "\n")
print("W的shape", W.shape)

print("\nX:\n")
print(X)

print("\nY:\n")
print(Y)

print("\nW:\n")
print(W)
print("\n", "-"*14, "end", "-"*14, "\n")










# print(x0)