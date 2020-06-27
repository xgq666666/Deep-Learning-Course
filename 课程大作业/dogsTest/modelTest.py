# VGG16模型
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 数据集处理
dir_path = "./images"

# 

i = 0


for fn in os.listdir(dir_path):
#     print(fn)
    
    dir_i = dir_path + "/" + fn + "/"
    filenames = tf.constant([dir_i + fn1 for fn1 in os.listdir(dir_i)])
    
#     print(filenames)
    
    if i==0 :
        filenames_all = tf.constant(filenames)   
        labels = tf.zeros(filenames.shape, dtype=tf.int32)
    else:
        filenames_all = tf.concat([filenames_all, filenames], axis=-1)
        labels_i = tf.fill(filenames.shape, i)
        labels = tf.concat([labels, labels_i], axis=-1)
        
    i = i + 1
    
dataset = tf.data.Dataset.from_tensor_slices((filenames_all, labels))

# 图像加载处理
def decode_image_and_resize(filename, label):
    image_string = tf.io.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_resized = tf.image.resize(image_decoded, [224,224]) / 255.0
    return image_resized, label

dataset = dataset.map(map_func=decode_image_and_resize,
                     num_parallel_calls=4)

# 打乱数据集
buffer_size = tf.size(labels)
buffer_size = tf.cast(buffer_size, dtype=tf.int64)
# labels.shape
# buffer_size
dataset = dataset.shuffle(buffer_size)

# 训练集选择
sub_dataset = dataset.take(10000)
batch_size = 10
sub_dataset = sub_dataset.batch(batch_size)
sub_dataset = sub_dataset.prefetch(tf.data.experimental.AUTOTUNE)

# 模型定义
def vgg16_model(input_shape=(224, 224, 3)):
    vgg16 = tf.keras.applications.vgg16.VGG16(include_top=False,
                                       weights='imagenet',
                                       input_shape=input_shape)
    
    for layer in vgg16.layers:
        layer.trainable = False
        
    last = vgg16.output
    
    # 加入全连接层
    x = tf.keras.layers.Flatten()(last)
    
#     x = tf.keras.layers.Dense(128, activation='relu')(x)
#     x = tf.keras.layers.Dropout(0.3)(x)
#     x = tf.keras.layers.Dense(128, activation='relu')(x)
#     x = tf.keras.layers.Dropout(0.3)(x)
    #输出层
    x = tf.keras.layers.Dense(120, activation='softmax')(x)
    
    #模型建立
    model = tf.keras.models.Model(inputs=vgg16.input, outputs=x)
    
    model.summary()
    
    return model

# 模型参数
model = vgg16_model()
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

training_epochs = 5

# 训练
model.fit(sub_dataset, epochs=training_epochs, verbose=1)


# 测试集
test_dataset = dataset.skip(10000)
batch_size = 10
test_dataset = test_dataset.batch(batch_size)
test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

# 保存模型
test_loss, test_acc = model.evaluate(test_dataset, verbose = 1)
model_filename = 'models/dogsCNNModel1-' + "%.6f-"%(test_loss) + "%.6f"%(test_acc) +'.h5'
model.save_weights(model_filename)
print("save ok!")

#模型结构保存
yaml_string = model.to_yaml()
with open('./models/dogs.yaml', 'w') as model_file:
    model_file.write(yaml_string)
    