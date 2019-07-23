# -*- coding: utf-8 -*-

import os
import tensorflow as tf
from PIL import Image
import numpy as np

cwd = 'data/'

img_classes = {'dark', 'light', 'motion', 'defocus', 'gaussian'}

def write(img_classes):
    writer = tf.python_io.TFRecordWriter("train.tfrecords")  # 要生成的文件
    
    for i in range(15600):
        print("NO." + str(i+1) + " is begin...")
        for j in range(2):
            label_path = "QR_make/label/NO" + str(i+1) + "_version" + str(j+1) + "_error25_label.png"
            for img_index, img_name in enumerate(img_classes):
            
                img_path = "QR_make/degraded/NO" + str(i+1) + "_version" + str(j+1) + "_error25_" + img_name + ".png"
                img = Image.open(img_path).convert('L')
                img_raw = img.tobytes()  # 将图片转化为二进制格式
                img_label = Image.open(label_path).convert('L')
                img_label = img_label.tobytes()
                features = tf.train.Features(
                    feature={"img_label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_label])),
                                 'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))})
    
                example = tf.train.Example(features=features)
                writer.write(example.SerializeToString()) 
                
            label_path = "QR_make/label/NO" + str(i+1) + "_version" + str(j+1) + "_error30_label.png"
            for img_index, img_name in enumerate(img_classes):
            
                img_path = "QR_make/degraded/NO" + str(i+1) + "_version" + str(j+1) + "_error30_" + img_name + ".png"
                img = Image.open(img_path).convert('L')
                img_raw = img.tobytes()  # 将图片转化为二进制格式
                img_label = Image.open(label_path).convert('L')
                img_label = img_label.tobytes()
                features = tf.train.Features(
                    feature={"img_label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_label])),
                                 'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))})
    
                example = tf.train.Example(features=features)
                writer.write(example.SerializeToString())      
                
            

    writer.close()
    print('Write tfrecords was done!')


def decode_from_tfrecords(file_name, bool_batch, batch_size=3):
    filename_queue = tf.train.string_input_producer([file_name])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件

    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'img_label': tf.FixedLenFeature([], tf.string),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })

    image = tf.decode_raw(features['img_raw'], tf.uint8)

    image_label = tf.decode_raw(features['img_label'], tf.uint8)

    # float16, float32, float64, int32, uint16, uint8, int16, int8, int64

    image = tf.reshape(image, [480, 480, 1])
    image_label = tf.reshape(image_label, [480, 480])

    image = tf.cast(image, tf.float32)
    image_label = tf.cast(image_label, tf.int32)

    if bool_batch:
        batch_size = batch_size
        min_after_dequeue = 10
        capacity = min_after_dequeue + 3 * batch_size

        # 随机打乱数据输入
        image, image_label = tf.train.shuffle_batch([image, image_label],
                                                    batch_size=batch_size,
                                                    num_threads=1,
                                                    capacity=capacity,
                                                    min_after_dequeue=min_after_dequeue)
    return image, image_label


if __name__ == "__main__":

    tf.reset_default_graph()

    file_name = "train.tfrecords"

    if True:
        write(img_classes)
    else:
        image, image_label = decode_from_tfrecords(file_name, bool_batch=False)

        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:  # 开始一个会话
            sess.run(init_op)
            # threads = tf.train.start_queue_runners(sess=sess)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            for i in range(6):
                example, l = sess.run([image, image_label])  # 在会话中取出image和label

                img_new = np.reshape(example, [480, 480])
                img_show = Image.fromarray(img_new).convert('L')
                img_show.save('example1' + str(i) + '.png')

                img_new = np.reshape(l, [480, 480])
                img_show = Image.fromarray(img_new).convert('L')
                img_show.save('example2' + str(i) + '.png')

"""

def read_test(input_file):

    # 用 dataset 读取 tfrecord 文件
    dataset = tf.data.TFRecordDataset(input_file)
    dataset = dataset.map(_parse_record)
    iterator = dataset.make_one_shot_iterator()


    with tf.Session() as sess:
        features = sess.run(iterator.get_next())
        name = features['name']
        name = name.decode()
        img_data = features['data']
        shape = features['shape']
        print('=======')
        print(type(shape))
        print(len(img_data))

        # 从 bytes 数组中加载图片原始数据，并重新 reshape.它的结果是 ndarray 数组
        img_data = np.fromstring(img_data,dtype=np.uint8)
        image_data = np.reshape(img_data,shape)


        plt.figure()
        #显示图片
        plt.imshow(image_data)
        plt.show()

        #将数据重新编码成 jpg 图片并保存
        img = tf.image.encode_jpeg(image_data)
        tf.gfile.GFile('cat_encode.jpg','wb').write(img.eval())

read_test('cat.tfrecord')

"""
















