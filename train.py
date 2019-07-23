# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np
from PIL import Image
import tfrecord
import os
from tensorflow.python.framework import graph_util
from skimage import io
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
kernel = 3
record_path = "train.tfrecords"


def get_weight(shape, gain=np.sqrt(2)):
    fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in)
    w = tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))
    return w


def upscale2d(x, factor=2):
    if factor == 1: return x
    with tf.variable_scope('Upscale2D'):
        s = x.shape
        x = tf.reshape(x, [-1, s[1], 1, s[2], 1, s[3]])
        x = tf.tile(x, [1, 1, factor, 1, factor, 1])
        x = tf.reshape(x, [-1, s[1] * factor, s[2] * factor, s[3]])
        return x


def network(x, width=480, height=480, num_channel=1):
    skips = [x]

    with tf.variable_scope('conv0'):
        w = get_weight([kernel, kernel, 1, 64])
        w = tf.cast(w, x.dtype)
        b = tf.get_variable('bias', shape=[64], initializer=tf.initializers.zeros())
        conv0 = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
        conv0_relu = tf.nn.leaky_relu(tf.nn.bias_add(conv0, b), alpha=0.1)

    with tf.variable_scope('conv1'):
        w = get_weight([kernel, kernel, 64, 64])
        b = tf.get_variable('bias', shape=[64], initializer=tf.initializers.zeros())
        conv1 = tf.nn.conv2d(conv0_relu, w, strides=[1, 1, 1, 1], padding='SAME')
        conv1_relu = tf.nn.leaky_relu(tf.nn.bias_add(conv1, b), alpha=0.1)

    maxpool0 = tf.nn.max_pool(conv1_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    skips.append(maxpool0)

    with tf.variable_scope('conv2'):
        w = get_weight([kernel, kernel, 64, 128])
        b = tf.get_variable('bias', shape=[128], initializer=tf.initializers.zeros())
        conv2 = tf.nn.conv2d(maxpool0, w, strides=[1, 1, 1, 1], padding='SAME')
        conv2_relu = tf.nn.leaky_relu(tf.nn.bias_add(conv2, b), alpha=0.1)

    maxpool1 = tf.nn.max_pool(conv2_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    skips.append(maxpool1)

    with tf.variable_scope('conv3'):
        w = get_weight([kernel, kernel, 128, 256])
        b = tf.get_variable('bias', shape=[256], initializer=tf.initializers.zeros())
        conv3 = tf.nn.conv2d(maxpool1, w, strides=[1, 1, 1, 1], padding='SAME')
        conv3_relu = tf.nn.leaky_relu(tf.nn.bias_add(conv3, b), alpha=0.1)

    maxpool2 = tf.nn.max_pool(conv3_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    skips.append(maxpool2)

    with tf.variable_scope('conv4'):
        w = get_weight([kernel, kernel, 256, 512])
        b = tf.get_variable('bias', shape=[512], initializer=tf.initializers.zeros())
        conv4 = tf.nn.conv2d(maxpool2, w, strides=[1, 1, 1, 1], padding='SAME')
        conv4_relu = tf.nn.leaky_relu(tf.nn.bias_add(conv4, b), alpha=0.1)

    maxpool3 = tf.nn.max_pool(conv4_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    skips.append(maxpool3)

    with tf.variable_scope('conv5'):
        w = get_weight([kernel, kernel, 512, 1024])
        b = tf.get_variable('bias', shape=[1024], initializer=tf.initializers.zeros())
        conv5 = tf.nn.conv2d(maxpool3, w, strides=[1, 1, 1, 1], padding='SAME')
        conv5_relu = tf.nn.leaky_relu(tf.nn.bias_add(conv5, b), alpha=0.1)

    maxpool4 = tf.nn.max_pool(conv5_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('conv6'):
        w = get_weight([kernel, kernel, 1024, 512])
        b = tf.get_variable('bias', shape=[512], initializer=tf.initializers.zeros())
        conv6 = tf.nn.conv2d(maxpool4, w, strides=[1, 1, 1, 1], padding='SAME')
        conv6_relu = tf.nn.leaky_relu(tf.nn.bias_add(conv6, b), alpha=0.1)

    # -------------------------------------------------------------------------------------------

    # up1 = upscale2d(conv6_relu)
    with tf.variable_scope('up1'):
        batch_size = tf.shape(conv6_relu)[0]
        deconv_shape = tf.stack([batch_size, 30, 30, 512])

        w = get_weight([kernel, kernel, 512, 512])
        up1 = tf.nn.conv2d_transpose(conv6_relu, w, output_shape=deconv_shape, strides=[1, 2, 2, 1], padding='SAME')

    dec_conv6 = tf.concat([up1, skips.pop()], axis=3)#30301024

    with tf.variable_scope('dec_conv5'):
        w = get_weight([kernel, kernel, 1024, 512])
        b = tf.get_variable('bias', shape=[512], initializer=tf.initializers.zeros())
        dec_conv5 = tf.nn.conv2d(dec_conv6, w, strides=[1, 1, 1, 1], padding='SAME')
        dec_conv5_relu = tf.nn.leaky_relu(tf.nn.bias_add(dec_conv5, b), alpha=0.1)

    with tf.variable_scope('dec_conv5b'):
        w = get_weight([kernel, kernel, 512, 256])
        b = tf.get_variable('bias', shape=[256], initializer=tf.initializers.zeros())
        dec_conv5b = tf.nn.conv2d(dec_conv5_relu, w, strides=[1, 1, 1, 1], padding='SAME')
        dec_conv5b_relu = tf.nn.leaky_relu(tf.nn.bias_add(dec_conv5b, b), alpha=0.1)

        # up2 = upscale2d(dec_conv5b_relu)
    with tf.variable_scope('up2'):
        batch_size = tf.shape(dec_conv5b_relu)[0]
        deconv_shape = tf.stack([batch_size, 60, 60, 256])

        w = get_weight([kernel, kernel, 256, 256])
        up2 = tf.nn.conv2d_transpose(dec_conv5b_relu, w, output_shape=deconv_shape, strides=[1, 2, 2, 1],
                                     padding='SAME')

    dec_conv5c = tf.concat([up2, skips.pop()], axis=3)

    with tf.variable_scope('dec_conv4'):
        w = get_weight([kernel, kernel, 512, 256])
        b = tf.get_variable('bias', shape=[256], initializer=tf.initializers.zeros())
        dec_conv4 = tf.nn.conv2d(dec_conv5c, w, strides=[1, 1, 1, 1], padding='SAME')
        dec_conv4_relu = tf.nn.leaky_relu(tf.nn.bias_add(dec_conv4, b), alpha=0.1)

    with tf.variable_scope('dec_conv4b'):
        w = get_weight([kernel, kernel, 256, 128])
        b = tf.get_variable('bias', shape=[128], initializer=tf.initializers.zeros())
        dec_conv4b = tf.nn.conv2d(dec_conv4_relu, w, strides=[1, 1, 1, 1], padding='SAME')
        dec_conv4b_relu = tf.nn.leaky_relu(tf.nn.bias_add(dec_conv4b, b), alpha=0.1)

        # up3 = upscale2d(dec_conv4b_relu)
    with tf.variable_scope('up3'):
        batch_size = tf.shape(dec_conv4b_relu)[0]
        deconv_shape = tf.stack([batch_size, 120, 120, 128])

        w = get_weight([kernel, kernel, 128, 128])
        up3 = tf.nn.conv2d_transpose(dec_conv4b_relu, w, output_shape=deconv_shape, strides=[1, 2, 2, 1],
                                     padding='SAME')

    dec_conv4c = tf.concat([up3, skips.pop()], axis=3)

    with tf.variable_scope('dec_conv3'):
        w = get_weight([kernel, kernel, 256, 128])
        b = tf.get_variable('bias', shape=[128], initializer=tf.initializers.zeros())
        dec_conv3 = tf.nn.conv2d(dec_conv4c, w, strides=[1, 1, 1, 1], padding='SAME')
        dec_conv3_relu = tf.nn.leaky_relu(tf.nn.bias_add(dec_conv3, b), alpha=0.1)

    with tf.variable_scope('dec_conv3b'):
        w = get_weight([kernel, kernel, 128, 64])
        b = tf.get_variable('bias', shape=[64], initializer=tf.initializers.zeros())
        dec_conv3b = tf.nn.conv2d(dec_conv3_relu, w, strides=[1, 1, 1, 1], padding='SAME')
        dec_conv3b_relu = tf.nn.leaky_relu(tf.nn.bias_add(dec_conv3b, b), alpha=0.1)

        # up4 = upscale2d(dec_conv3b_relu)
    with tf.variable_scope('up4'):
        batch_size = tf.shape(dec_conv3b_relu)[0]
        deconv_shape = tf.stack([batch_size, 240, 240, 64])

        w = get_weight([kernel, kernel, 64, 64])
        up4 = tf.nn.conv2d_transpose(dec_conv3b_relu, w, output_shape=deconv_shape, strides=[1, 2, 2, 1],
                                     padding='SAME')

    dec_conv3c = tf.concat([up4, skips.pop()], axis=3)

    with tf.variable_scope('dec_conv2'):
        w = get_weight([kernel, kernel, 128, 64])
        b = tf.get_variable('bias', shape=[64], initializer=tf.initializers.zeros())
        dec_conv2 = tf.nn.conv2d(dec_conv3c, w, strides=[1, 1, 1, 1], padding='SAME')
        dec_conv2_relu = tf.nn.leaky_relu(tf.nn.bias_add(dec_conv2, b), alpha=0.1)

    with tf.variable_scope('dec_conv2b'):
        w = get_weight([kernel, kernel, 64, 64])
        b = tf.get_variable('bias', shape=[64], initializer=tf.initializers.zeros())
        dec_conv2b = tf.nn.conv2d(dec_conv2_relu, w, strides=[1, 1, 1, 1], padding='SAME')
        dec_conv2b_relu = tf.nn.leaky_relu(tf.nn.bias_add(dec_conv2b, b), alpha=0.1)

        # up5 = upscale2d(dec_conv2b_relu)
    with tf.variable_scope('up5'):
        batch_size = tf.shape(dec_conv2b_relu)[0]
        deconv_shape = tf.stack([batch_size, 480, 480, 64])

        w = get_weight([kernel, kernel, 64, 64])
        up5 = tf.nn.conv2d_transpose(dec_conv2b_relu, w, output_shape=deconv_shape, strides=[1, 2, 2, 1],
                                     padding='SAME')

    dec_conv2c = tf.concat([up5, skips.pop()], axis=3)

    with tf.variable_scope('dec_conv1'):
        w = get_weight([kernel, kernel, 65, 65])
        b = tf.get_variable('bias', shape=[65], initializer=tf.initializers.zeros())
        dec_conv1 = tf.nn.conv2d(dec_conv2c, w, strides=[1, 1, 1, 1], padding='SAME')
        dec_conv1_relu = tf.nn.leaky_relu(tf.nn.bias_add(dec_conv1, b), alpha=0.1)

    with tf.variable_scope('dec_conv1b'):
        w = get_weight([kernel, kernel, 65, 32])
        b = tf.get_variable('bias', shape=[32], initializer=tf.initializers.zeros())
        dec_conv1b = tf.nn.conv2d(dec_conv1_relu, w, strides=[1, 1, 1, 1], padding='SAME')
        dec_conv1b_relu = tf.nn.leaky_relu(tf.nn.bias_add(dec_conv1b, b), alpha=0.1)

    with tf.variable_scope('dec_conv0'):
        w = get_weight([kernel, kernel, 32, 2])
        b = tf.get_variable('bias', shape=[2], initializer=tf.initializers.zeros())
        dec_conv0 = tf.nn.conv2d(dec_conv1b_relu, w, strides=[1, 1, 1, 1], padding='SAME')
        dec_conv0_final = tf.nn.bias_add(dec_conv0, b, name="denoised")

    return dec_conv0_final


def text_save(filename, data):
    file = open(filename, 'w')
    for i in range(len(data)):
        if i != len(data) - 1:
            s = str(data[i]) + ", "
            file.write(s)
        else:
            s = str(data[i])
            file.write(s)
    file.close()
    print("The data is saved!\n")


if __name__ == "__main__":

    epoch = 10
    disp = 5

    learning_rate_list = [0.001, 0.0005, 0.0001]
    batch_size_list = [8, 16, 32]

    index = 1
    modelnumber = 1

    for m in range(1):
        for n in range(1):

            tf.reset_default_graph()

            loss_result = []

            batch_size = 16  # batch_size_list[m]
            learning_rate = 0.0003  # learning_rate_list[n]

            print("\n")
            print("The batch size is: " + str(batch_size))
            print("The learning rate is: " + str(learning_rate))
            print("The train begins!\n")
            print("\n")

            image, image_label = tfrecord.decode_from_tfrecords(record_path, bool_batch=True, batch_size=batch_size)

            #with tf.name_scope('Inputs'), tf.device("/cpu:0"):

            x_Inputs = tf.placeholder(tf.float32, name='x_Inputs', shape=[None, 480, 480, 1])
            y_label = tf.placeholder(tf.int32, name='y_label', shape=[None, 480, 480])

            #with tf.device("/gpu:0"):

            denoised = network(x_Inputs, width=480, height=480, num_channel=1)

            with tf.name_scope('softmax_loss'):
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_label, logits=denoised, name='loss')
                loss_mean = tf.reduce_mean(loss)

            with tf.name_scope('Gradient_Descent'):
                train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=loss_mean)

            # graph = tf.Graph()
            # print(tf.get_default_graph().as_graph_def().node)
            init_op = tf.global_variables_initializer()
            # saver = tf.train.Saver()

            with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

                sess.run(init_op)

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                for i in range(epoch):

                    print("The " + str(i) + ": epoch begins")
                    print("\t loading...")
                    example, label = sess.run([image, image_label])  # 在会话中取出image和label

                    l, _ = sess.run((loss_mean, train_step), feed_dict={x_Inputs: example, y_label: label})

                    print("The " + str(i) + ": epoch ends")
                    print("\n")

                    # Dump training status
                    if i % disp == 0:

                        print("\tThe No." + str(i) + " loss is: " + str(l))
                        print("\n")
                        loss_result.append(l)

                        output_graph = "saved_model/model" + str(modelnumber) + "/model.pb"
                        if not os.path.exists("saved_model/model" + str(modelnumber)):
                            os.makedirs("saved_model/model" + str(modelnumber))

                        graph_def = tf.get_default_graph().as_graph_def()
                        output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def,
                                                                                     ["dec_conv0/denoised"])
                        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
                            f.write(output_graph_def.SerializeToString())  # 序列化输出
                        print("model saved!\n")

                        # 中途打印信息
                        data_img = Image.open('data/1.png').convert('L')
                        test_img = Image.open('data/2.png').convert('L')

                        data_img = np.reshape(data_img, [1, 128, 128, 1])
                        test_img = np.reshape(test_img, [1, 128, 128, 1])

                        batch_size = 1

                        denoised_data_img = sess.run(tf.argmax(input=denoised, axis=3), feed_dict={x_Inputs: data_img})
                        denoised_test_img = sess.run(tf.argmax(input=denoised, axis=3), feed_dict={x_Inputs: test_img})

                        denoised_data_img = np.array(denoised_data_img)
                        denoised_test_img = np.array(denoised_test_img)

                        denoised_data_img = np.reshape(denoised_data_img, [128, 128])
                        denoised_test_img = np.reshape(denoised_test_img, [128, 128])

                        for rows in range(128):
                            for cols in range(128):
                                    denoised_data_img[rows][cols] = denoised_data_img[rows][cols] * 255
                                    denoised_test_img[rows][cols] = denoised_test_img[rows][cols] * 255


                        #denoised_data_img = Image.fromarray(denoised_data_img)
                        #denoised_test_img = Image.fromarray(denoised_test_img)

                        saved_data_path = "result/image/data/" + str(index) + "/"
                        saved_test_path = "result/image/test/" + str(index) + "/"

                        if not os.path.exists(saved_data_path):
                            os.makedirs(saved_data_path)

                        if not os.path.exists(saved_test_path):
                            os.makedirs(saved_test_path)

                        #denoised_data_img.save("result/image/data/" + str(index) + "/result" + str(i) + ".png")
                        #denoised_test_img.save("result/image/test/" + str(index) + "/result" + str(i) + ".png")

                        io.imsave("result/image/data/" + str(index) + "/result" + str(i) + ".png", denoised_data_img)
                        io.imsave("result/image/test/" + str(index) + "/result" + str(i) + ".png", denoised_test_img)

                        modelnumber = modelnumber + 1

                text_save("result/loss" + str(index) + ".txt", loss_result)
                index = index + 1
                coord.request_stop()
                coord.join(threads)
                print("Done!")





