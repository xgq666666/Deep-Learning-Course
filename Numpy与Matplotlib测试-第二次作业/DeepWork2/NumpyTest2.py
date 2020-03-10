import numpy as np

x = np.array([64.3, 99.6, 145.45, 63.75, 135.46, 92.85, 86.97, 144.76, 59.3, 116.03])

y = np.array([62.55, 82.42, 132.62, 73.31, 131.05, 86.57, 85.49, 127.44, 55.25, 104.84])

#x,y的均值
xm = np.mean(x)
ym = np.mean(y)

#计算w
s1 = np.sum(((x - xm) * (y - ym)))
s2 = np.sum(np.power((x - xm), 2))
w = s1 / s2

#计算b
b = ym - (w * xm)

print("\n", "-"*13, "start", "-"*13, "\n")
print("W = ", w)
print("b = ", b)
print("\n", "-"*14, "end", "-"*14, "\n")



# print(s2)



# x3 = np.power((x - np.mean(x)), 2)

# print(x3)


# x2 = (x - np.mean(x)) * (y - np.mean(y))

# print(x2)

# x1 = x - np.mean(x)

# print(x1)

# s1 = np.mean(x)

# s2 = np.sum(x) / len(x)

# print(s1)

# print(s2)

# print(len(x))

# print(len(y))