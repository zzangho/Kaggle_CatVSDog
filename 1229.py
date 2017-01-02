import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy import misc
import os
import tensorflow as tf


def init_weights(shape, vname):
    return tf.Variable(initial_value = tf.random_normal(shape, stddev=0.01), name=vname)

def im_resize_pad(image, target_size): # to square image
    im_shape = image.shape

    if im_shape[0] > im_shape[1]: # height is longer than weight
        # Resize maintaining aspect ratio
        pad_dim = im_shape[1]

        image = misc.imresize( image, [target_size, int( (im_shape[1]/im_shape[0])*target_size )]  )
        rest = target_size - image.shape[1]

        # Pad rest of part
        one_side = int(rest / 2)
        other_side = rest - one_side

        image = np.pad(image, ((0, 0), (one_side, other_side), (0, 0)), 'constant')
    else:
        # Resize maintaining aspect ratio
        pad_dim = im_shape[0]

        image = misc.imresize( image, [int( (im_shape[0] / im_shape[1]) * target_size ), target_size] )
        rest = target_size - image.shape[0]

        # Pad rest of part
        one_side = int(rest / 2)
        other_side = rest - one_side

        image = np.pad(image, ((one_side, other_side), (0, 0), (0, 0)), 'constant')

#    plt.figure()
#    plt.imshow(image)
#    plt.show()

    return image


def mdl(ws, p_keep_conv, p_keep_hidden, X):
    l1a = tf.nn.relu(tf.nn.conv2d(X, ws[0],  # l1a shape=(?, 28, 28, 32)
                                  strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],  # l1 shape=(?, 14, 14, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, ws[1],  # l2a shape=(?, 14, 14, 64)
                                  strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],  # l2 shape=(?, 7, 7, 64)
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, ws[2],  # l3a shape=(?, 7, 7, 128)
                                  strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],  # l3 shape=(?, 4, 4, 128)
                        strides=[1, 2, 2, 1], padding='SAME')

    l4a = tf.nn.relu(tf.nn.conv2d(l3, ws[3],  # l3a shape=(?, 7, 7, 128)
                                  strides=[1, 1, 1, 1], padding='SAME'))
    l4 = tf.nn.max_pool(l4a, ksize=[1, 2, 2, 1],  # l3 shape=(?, 4, 4, 128)
                        strides=[1, 2, 2, 1], padding='SAME')
    l4 = tf.nn.dropout(l4, p_keep_conv)

    l5a = tf.nn.relu(tf.nn.conv2d(l4, ws[4],  # l3a shape=(?, 7, 7, 128)
                                  strides=[1, 1, 1, 1], padding='SAME'))
    l5 = tf.nn.max_pool(l5a, ksize=[1, 2, 2, 1],  # l3 shape=(?, 4, 4, 128)
                        strides=[1, 2, 2, 1], padding='SAME')
    l5 = tf.reshape(l5, [-1, ws[5].get_shape().as_list()[0]])  # reshape to (?, 2048)


    l6 = tf.nn.relu(tf.matmul(l5, ws[5]))
    l6 = tf.nn.dropout(l6, p_keep_hidden)

    pyx = tf.matmul(l6, ws[6])
    return pyx

def get_batch_image_label(filenames):
    images = np.empty((len(filenames), 400, 400, 3))
    labels = np.zeros( (len(filenames), 2) )

    for idx,filename in enumerate(filenames):

        image = misc.imread(filename)
        #image = misc.imresize( image, [400, 400])
        image = im_resize_pad(image, 400)

        images[idx,:,:,:] = image

        name_set = filename.split('/')[-1]
        name = name_set.split('.')[0]

        if name=='cat':
            labels[idx,0] = 1
        else:
            labels[idx,1] = 1

    return images, labels

################################################################
# Get List of TEST of TRAIN Paths

train_paths = []
test_paths = []

TRAIN_DIR = 'input/train/'
TEST_DIR = 'input/test/'
train_imnames = os.listdir(TRAIN_DIR)
test_imnames = os.listdir(TEST_DIR)

for train_imname in train_imnames:
    train_paths.append(os.path.join(TRAIN_DIR, train_imname))

for test_imname in train_imnames:
    test_paths.append(os.path.join(TEST_DIR, test_imname))

w1 = init_weights([10, 10, 3, 32], vname='w1')       # 3x3x1 conv, 32 outputs
w2 = init_weights([8, 8, 32, 64], vname='w2')     # 3x3x32 conv, 64 outputs
w3 = init_weights([8, 8, 64, 128], vname='w3')    # 3x3x32 conv, 128 outputs
w4 = init_weights([8, 8, 128, 128], vname='w4')    # 3x3x32 conv, 128 outputs
w5 = init_weights([8, 8, 128, 128], vname='w5')    # 3x3x32 conv, 128 outputs
w6 = init_weights([128 * 13 * 13, 625], vname='w6') # FC 128 * 4 * 4 inputs, 625 outputs
wo = init_weights([625, 2], vname='wo')         # FC 625 inputs, 10 outputs (labels)
p_keep_conv = tf.constant(.8)
p_keep_hidden = tf.constant(.5)



batch_size = 10

X = tf.placeholder( tf.float32, [None, 400, 400, 3] )
Y = tf.placeholder( tf.float32, [None, 2] )
py_x = mdl([w1, w2, w3, w4, w5, w6, wo], p_keep_conv, p_keep_hidden, X)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
#train_op = tf.train.AdamOptimizer(0.001, 0.90).minimize(cost)
train_op = tf.train.RMSPropOptimizer(0.001,1).minimize(cost)
predict_op = tf.argmax(py_x, 1)
test_op = tf.argmax(Y,1)
test_accuracy = tf.reduce_mean( tf.cast( tf.equal(predict_op, test_op), tf.float32 ) )

#######################################

# Start a new session to show example output.
saver = tf.train.Saver(max_to_keep = 50)


with tf.Session() as sess:
    # Coordinate the loading of image files.
    tf.global_variables_initializer().run()

    for i in range(50):
        # Make Batch of Image
        np.random.shuffle(train_paths)

        saver.save(sess, "tmp/model.chk", global_step=i)

        for j in range( batch_size, len(train_paths), batch_size ):
            train_paths_part = train_paths[j-batch_size:j]

            train_images_part, train_labels_part = get_batch_image_label(train_paths_part)
            sess.run(train_op, feed_dict={X: train_images_part, Y: train_labels_part})

            if j%100 == 0:
                print('epoch',i,'iter:',j,':',sess.run( [test_accuracy,cost], feed_dict={X: train_images_part, Y:train_labels_part } ))


