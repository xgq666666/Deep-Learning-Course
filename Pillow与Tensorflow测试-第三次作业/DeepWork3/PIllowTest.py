import matplotlib.pyplot as plt
from PIL import Image

#初始
plt.rcParams['font.sans-serif'] = ["SimHei"]
#打开图片并进行通道分离
# img = Image.open("D:\\code\\python\\DeepWork3\\lena.tiff")
img = Image.open("lena.tiff")
img.thumbnail((250, 250))   #像素缩放到与题目所显示的一致便于对比
img_r, img_g, img_b = img.split()

#R通道缩放为50*50
img_rr = img_r.resize((50, 50))

#G通道水平镜像-顺时针旋转90度
img_gt = img_g.transpose(Image.FLIP_LEFT_RIGHT).rotate(-90)

#B通道裁剪
img_bc = img_b.crop((0, 0, 150, 150))

#合并分离的通道
img_m = Image.merge("RGB", [img_r, img_g, img_b])
img_m.save("test.png")  #按png格式保存合并的图片

#画图显示
plt.figure(num = "xgq")

plt.subplot(221)
plt.imshow(img_rr, cmap="gray")
plt.axis("off")
plt.title("R-缩放", fontsize=14)

plt.subplot(222)
plt.imshow(img_gt, cmap="gray")
plt.title("G-镜像+旋转", fontsize=14)

plt.subplot(223)
plt.imshow(img_bc, cmap="gray")
plt.axis("off")
plt.title("B-裁剪", fontsize=14)

plt.subplot(224)
plt.imshow(img_m)
plt.axis("off")
plt.title(img_m.mode, fontsize=14)

plt.tight_layout(rect = [0, 0, 1, 0.9])
plt.suptitle("图像基本操作", fontsize=20, color='blue')
plt.show()

