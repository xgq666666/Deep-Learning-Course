import tkinter as tk
from PIL import ImageTk
from PIL import Image
from tkinter.filedialog import askdirectory
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 选择图片
def callback():
    fileName =tk.filedialog.askopenfilename(filetypes=[("JPG",".jpg"),("PNG",".png")])
    print(fileName)
    img = Image.open(fileName)
    imgg = img.resize((224, 224))
    imggg = imgg.convert('RGB')
    imggg.save('dog.jpg')
    photo = ImageTk.PhotoImage(file='dog.jpg')
    # photo = Image.open(fileName)
    print(photo)
    # img_label = tk.Label(root, imag=photo)
    img_label.config(imag=photo)
    # img_label.pack()
    tk.mainloop()

# 字符串处理
def stringE(s):
    return s[10:]


# 预测图片
def predDog():
    img = Image.open('dog.jpg')
    img_r = img.resize((224, 224))
    img_data = np.array(img_r)
    img_data2 = np.array([img_data])

    # 进行预测
    preds = model.predict(img_data2)

    i = tf.argmax(preds, 1).numpy()[0]

    dir_path = './images'

    # print(i)
    dir_path = "./images"
    f_l = [fn2 for fn2 in os.listdir(dir_path)]

    s = stringE(f_l[i])

    txt_label.config(text=s)

# 模型加载
with open('./models/dogs.yaml') as yamlfile:
    loaded_model_yaml = yamlfile.read()
model = tf.keras.models.model_from_yaml(loaded_model_yaml)

# 模型权重参数
model.load_weights('./models/dogsCNNModel1-1.771199-0.734026.h5')

# 显示界面
root= tk.Tk()
root.title('预测实现')
root.geometry('300x400') 
img = Image.open('dog.jpg')
imgg = img.resize((224, 224))
imggg = imgg.convert('RGB')
imggg.save('dog.jpg')
photo = ImageTk.PhotoImage(file='dog.jpg')
img_label = tk.Label(root, imag=photo)
img_label.pack()


b = tk.Button(root, 
    text='选择图片',      # 显示按钮上的文字
    width=15, height=2, 
    command=callback)     # 点击按钮执行的命令
b.pack()                # 按钮位置


b1 = tk.Button(root, 
    text='预测图片',      # 显示按钮上的文字
    width=15, height=2, 
    command=predDog)     # 点击按钮执行的命令
b1.pack()  

txt_label = tk.Label(root, text='选择小狗jpg图片进行预测...预测经供参考OvO')
txt_label.pack()
root.mainloop()