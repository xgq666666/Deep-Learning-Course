import numpy as np

#随机数产生
np.random.seed(612)
a = np.random.uniform(0, 1, 1000)

# print(len(a))


x = int(input("请输入一个(1-100)之间的整数："))

# print(x)


#直接使用x的倍数进行判断输出
i = 0
t = i * x

print("\n", "-"*13, "start", "-"*13, "\n")
print("序号\t索引值\t随机数")

while t < 1000:
    print(i+1, '\t', t, '\t', a[t])
    i += 1
    t = i * x

print("\n", "-"*14, "end", "-"*14, "\n")