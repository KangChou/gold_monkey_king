import os
import glob
import time
import numpy as np
import tensorflow as tf
from skimage import io, transform

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'
# 这是默认的显示等级，显示所有信息
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
# 只显示 warning 和 Error
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
# 只显示 Error


# 读取图片
def read_img(path, w, h):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    # print(cate)

    imgs = []
    labels = []

    print('Start read the image ...')

    for index, folder in enumerate(cate):
        # print(index, folder)
        for im in glob.glob(folder + '/*.jpg'):
            # print('Reading The Image: %s' % im)
            img = io.imread(im)
            img = transform.resize(img, (w, h))
            imgs.append(img)
            # print(index)
            labels.append(index)                    # 每个子文件夹是一个label
            # print(len(labels))

    print('Finished ...')
    print(len(imgs))
    print(len(labels))

    return np.asarray(imgs, np.float32), np.asarray(labels, np.float32)


# 打乱顺序 为了均匀取数据
def messUpOrder(data, label):
    num_example = data.shape[0]
    arr = np.arange(num_example)
    np.random.shuffle(arr)
    data = data[arr]
    label = label[arr]

    return data, label


# 将所有数据分为训练集和验证集
def segmentation(data, label, ratio=0.8):
    num_example = data.shape[0]
    s = np.int(num_example * ratio)
    x_train = data[:s]
    y_train = label[:s]
    x_val = data[s:]
    y_val = label[s:]

    return x_train, y_train, x_val, y_val


# 构建网络
def buildCNN(w, h, c):
    # 占位符
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, shape=[None, w, h, c], name='x-input')
        y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')

    # 第一个卷积层 + 池化层（100——>50)
    with tf.name_scope('conv2d-pool-1'):
        conv1 = tf.layers.conv2d(
            inputs=x,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # 第二个卷积层 + 池化层 (50->25)
    with tf.name_scope('conv2d-pool-2'):
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # 第三个卷积层 + 池化层 (25->12)
    with tf.name_scope('conv2d-pool-3'):
        conv3 = tf.layers.conv2d(
            inputs=pool2,
            filters=128,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    # 第四个卷积层 + 池化层 (12->6)
    with tf.name_scope('conv2d-pool-4'):
        conv4 = tf.layers.conv2d(
            inputs=pool3,
            filters=128,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
        pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    re1 = tf.reshape(pool4, [-1, 6 * 6 * 128])

    # 全连接层
    with tf.name_scope('layer-1'):
        dense1 = tf.layers.dense(inputs=re1,
                                 units=1024,
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    with tf.name_scope('layer-2'):
        dense2 = tf.layers.dense(inputs=dense1,
                                 units=512,
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    with tf.name_scope('layer-3'):
        dense3 = tf.layers.dense(inputs=dense2,
                                 units=32,
                                 activation=None,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    with tf.name_scope('layer-4'):
        logits = tf.layers.dense(inputs=dense3,
                                 units=7,
                                 activation=None,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    # 定义最后一层以此（此层只是用来挤压最后一个全连接层，得到概率，这个层次并未在后面的优化器中使用，只是为了）
    #为了得到pb文件，或者在tensorboard可视化，以及用于导入OpenCV中使用
    output = tf.nn.softmax(logits, name='output')
    return logits, x, y_


# 返回损失函数的值，准确值等参数
def accCNN(logits, y_):
    with tf.name_scope('cross_entropy'):
        loss = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=logits)
    with tf.name_scope('AdamOptimizer'):
        train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)   #优化器
    with tf.name_scope('acc'):
        correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return loss, train_op, correct_prediction, acc


# 定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets), 'len(inputs) != len(targets)'
    if shuffle:
        indices = np.arange(len(inputs))   # [0, 1, 2, ..., 422]
        np.random.shuffle(indices)     # 打乱下标顺序
    #从0开始，到结束，步长为batch_size
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

# 训练和测试（加入模型加载及保存功能，使其为增量训练）
def runable(x_train, y_train, train_op, loss, acc, x, y_, x_val, y_val):
    # 训练和测试数据，可将n_epoch设置更大一些
    n_epoch = 10              #之前是100轮，有点多，改成增量训练模型后就可以多轮次增加训练了
    batch_size = 30             #20是每批读取数据的数量

    ###创建saver对象用于模型保存和加载
    saver = tf.train.Saver()

    sess = tf.InteractiveSession()                          #手动启动session对话，还需关闭（也可使用with）
    sess.run(tf.global_variables_initializer())             #对全局变量初始化先

    ###训练之前，检查是否存在已经训练的模型（要在初始化后进行判断）
    # 如果存在则加载进来，进行增量训练
    if os.path.exists("./model/checkpoint"):  # 判断此文件是否存在
        saver.restore(sess, "./model/")  # 将此模型加载到sess中，sess就有了经过训练的参数了，如权重和偏执，经过梯度下降训练了一段时间的
        print("模型加载成功")

    for epoch in range(n_epoch):
        print('epoch: ', epoch)             #打印轮次
        # training
        train_loss, train_acc, n_batch = 0, 0, 0
        for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):        #定义每轮的训练批次
            # print('x_val_a: ', x_val_a.shape())
            # print('y_val_a: ', y_val_a.shape())
            _, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y_: y_train_a}) #执行操作并返回参数
            train_loss += err
            train_acc += ac
            n_batch += 1
        print("train loss: %f" % (train_loss / n_batch))
        print("train acc: %f" % (train_acc / n_batch))

        # validation
        val_loss, val_acc, n_batch = 0, 0, 0
        for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
            # print('x_val_a: ', x_val_a)
            # print('y_val_a: ', y_val_a)
            err, ac = sess.run([loss, acc], feed_dict={x: x_val_a, y_: y_val_a})
            val_loss += err
            val_acc += ac
            n_batch += 1
        print("validation loss: %f" % (val_loss / n_batch))
        print("validation acc: %f" % (val_acc / n_batch))
        print('*' * 50)

    ###训练完成后保存模型
    saver.save(sess, "./model/")

    sess.close()            #关闭会话


if __name__ == '__main__':
    imgpath = 'dataset/'

    w = 100
    h = 100
    c = 3

    ratio = 0.8  # 选取训练集的比例

    data, label = read_img(path=imgpath, w=w, h=h)#读取图片
    print(0)

    data, label = messUpOrder(data=data, label=label)#打乱顺序
    print(1)

    x_train, y_train, x_val, y_val = segmentation(data=data, label=label, ratio=ratio)  #分为训练集和验证集
    print('x_train: ', len(x_train))
    print('y_train: ', len(y_train))
    print(2)

    logits, x, y_ = buildCNN(w=w, h=h, c=c)     #构建网络返回最后一层结果，此时还不执行，只是操作
    print(3)
    print(logits)

    loss, train_op, correct_prediction, acc = accCNN(logits=logits, y_=y_)      #评估网络，返回损失函数，训练，预测值，准确率
    #返回损失函数，优化器，预测值，准确率等操作，需在后面的训练函数中执行操作
    print(4)

    runable(x_train=x_train, y_train=y_train, train_op=train_op, loss=loss, acc=acc, x=x, y_=y_, x_val=x_val, y_val=y_val)
