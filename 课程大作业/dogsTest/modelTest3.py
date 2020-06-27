# DenseNet121模型
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
    filenames = [dir_i + fn1 for fn1 in os.listdir(dir_i)]
    
#     print(filenames)
    
    if i==0 :
        filenames_all = filenames
        labels = [i] * len(filenames)
    else:
        filenames_all = filenames_all + filenames
        labels_i = [i] * len(filenames)
        labels = labels + labels_i
        
    i = i + 1


# TFRecord
def write_TFRecord_file(filenames_all, labels, tfrecord_file):
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        for filename, label in zip(filenames_all, labels):
#             print(filename)
            image = open(filename, 'rb').read()
            
            feature = {
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }
            
            example = tf.train.Example(features=tf.train.Features(feature=feature))
#             print(example.SerializeToString)
            writer.write(example.SerializeToString())


feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64)
}

# 图像处理
def parse_example(example_string):
    feature_dict = tf.io.parse_single_example(example_string, feature_description)
    feature_dict['image'] = tf.io.decode_jpeg(feature_dict['image'], channels=3)
    feature_dict['image'] = tf.image.resize(feature_dict['image'], [224, 224]) / 255.0
    return feature_dict['image'], feature_dict['label']


# 数据集加载
data_dir = './data/'
tfrecord_file = data_dir + 'dogs.tfrecords'

if not os.path.isfile(tfrecord_file):
    write_TFRecord_file(filenames_all, labels, tfrecord_file)
    print('write TFRecord file:', tfrecord_file)
else:
    print(tfrecord_file, 'already exists')

def read_TFRecord_file(tfrecord_file):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
    
    dataset = raw_dataset.map(parse_example)
    
    return dataset

dataset = read_TFRecord_file(tfrecord_file)

# 打乱数据集
buffer_size = tf.size(labels)
buffer_size = tf.cast(buffer_size, dtype=tf.int64)
# labels.shape
# buffer_size
dataset = dataset.shuffle(buffer_size)

# 训练集
train_dataset = dataset.take(10000)

batch_size = 10
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

# 模型定义
def DenseNet121_model(input_shape=(224, 224, 3)):
    md = tf.keras.applications.densenet.DenseNet121(include_top=False,
                                       weights='imagenet',
                                       input_shape=input_shape)
    
    for layer in md.layers:
        layer.trainable = False
        
    last = md.output
    
    # 加入全连接层
    x = tf.keras.layers.Flatten()(last)
    
#     x = tf.keras.layers.Dense(128, activation='relu')(x)
#     x = tf.keras.layers.Dropout(0.3)(x)
#     x = tf.keras.layers.Dense(128, activation='relu')(x)
#     x = tf.keras.layers.Dropout(0.3)(x)
    #输出层
    x = tf.keras.layers.Dense(120, activation='softmax')(x)
    
    #模型建立
    model = tf.keras.models.Model(inputs=md.input, outputs=x)
    
    model.summary()
    
    return model

# 模型参数
model = DenseNet121_model()

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

training_epochs = 5

# 训练
history = model.fit(train_dataset, epochs=training_epochs, verbose=1)


# 测试集
test_dataset = dataset.skip(10000)

batch_size = 10
test_dataset = test_dataset.batch(batch_size)
test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)

# 保存模型
test_loss, test_acc = model.evaluate(test_dataset, verbose = 1)
model_filename = 'models/DenseNet121-dogsCNNModel1-' + "%.6f-"%(test_loss) + "%.6f"%(test_acc) +'.h5'
model.save_weights(model_filename)
print("save ok!")

#模型结构保存
yaml_string = model.to_yaml()
with open('./models/DenseNet121-dogs.yaml', 'w') as model_file:
    model_file.write(yaml_string)