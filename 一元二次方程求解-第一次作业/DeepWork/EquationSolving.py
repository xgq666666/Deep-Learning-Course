#一元二次方程求解


a, b, c =map(int,input('输入a,b,c的值(空格隔开):').split())     

#利用判断式进行判断
d = (b ** 2 ) - (4 * a * c)

if d < 0 :
    print("无解")
else :
    if d == 0 :
        x = -(b / (2 * a))
        print("解为：x={}".format(x))
    else:
        x1 = ((d ** 0.5) - b) / (2 * a)
        x2 = (-(d ** 0.5) - b) / (2 * a)
        print("解为：x1={} ,x2={}".format(x1, x2))


#print(d)