import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy import misc
import os
import tensorflow as tf


def init_weights(shape):
    return tf.Variable(initial_value = tf.random_normal(shape, stddev=0.01))

class Layers:
    def __init__(self, w1, w2, w3, w4, wo, p_keep_conv, p_keep_hidden):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.wo = wo

        self.p_keep_conv = p_keep_conv
        self.p_keep_hidden = p_keep_hidden

    def esimate(self, X):
        l1a = tf.nn.relu(tf.nn.conv2d(X, self.w1,  # l1a shape=(?, 28, 28, 32)
                                      strides=[1, 1, 1, 1], padding='SAME'))
        l1 = tf.nn.max_pool(l1a, ksize=[1, 4, 4, 1],  # l1 shape=(?, 14, 14, 32)
                            strides=[1, 4, 4, 1], padding='SAME')
        l1 = tf.nn.dropout(l1, self.p_keep_conv)

        l2a = tf.nn.relu(tf.nn.conv2d(l1, self.w2,  # l2a shape=(?, 14, 14, 64)
                                      strides=[1, 1, 1, 1], padding='SAME'))
        l2 = tf.nn.max_pool(l2a, ksize=[1, 4, 4, 1],  # l2 shape=(?, 7, 7, 64)
                            strides=[1, 4, 4, 1], padding='SAME')
        l2 = tf.nn.dropout(l2, self.p_keep_conv)

        l2a = tf.nn.relu(tf.nn.conv2d(l1, self.w2,  # l2a shape=(?, 14, 14, 64)
                                      strides=[1, 1, 1, 1], padding='SAME'))
        l2 = tf.nn.max_pool(l2a, ksize=[1, 4, 4, 1],  # l2 shape=(?, 7, 7, 64)
                            strides=[1, 4, 4, 1], padding='SAME')
        l2 = tf.nn.dropout(l2, self.p_keep_conv)

        l2a = tf.nn.relu(tf.nn.conv2d(l1, self.w2,  # l2a shape=(?, 14, 14, 64)
                                      strides=[1, 1, 1, 1], padding='SAME'))
        l2 = tf.nn.max_pool(l2a, ksize=[1, 4, 4, 1],  # l2 shape=(?, 7, 7, 64)
                            strides=[1, 4, 4, 1], padding='SAME')
        l2 = tf.nn.dropout(l2, self.p_keep_conv)

        l5a = tf.nn.relu(tf.nn.conv2d(l2, self.w3,  # l3a shape=(?, 7, 7, 128)
                                      strides=[1, 1, 1, 1], padding='SAME'))
        l5 = tf.nn.max_pool(l3a, ksize=[1, 4, 4, 1],  # l3 shape=(?, 4, 4, 128)
                            strides=[1, 4, 4, 1], padding='SAME')

        l5 = tf.reshape(l3, [-1, self.w6.get_shape().as_list()[0]])  # reshape to (?, 2048)
        l5 = tf.nn.dropout(l3, self.p_keep_conv)

        l6 = tf.nn.relu(tf.matmul(l3, self.w6))
        l6 = tf.nn.dropout(l4, self.p_keep_hidden)

        pyx = tf.matmul(l4, self.wo)
        return pyx

def get_batch_image_label(filenames):
    images = np.empty((len(filenames),300,300,3))
    labels = []

    for idx,filename in enumerate(filenames):
#        file_content = tf.read_file(filename)
#        image = tf.image.decode_jpeg( file_content, channels=3 )
#        image = tf.image.resize_image_with_crop_or_pad(image, 400, 400)

        image = misc.imread(filename)
        image = misc.imresize( image, [300, 300])

        images[idx,:,:,:] = image

        name_set = filename.split('/')[-1]
        name = name_set.split('.')[0]

        if name=='cat':
            labels.append([1, 0])
        else:
            labels.append([0, 1])

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

w1 = init_weights([10, 10, 3, 32])       # 3x3x1 conv, 32 outputs
w2 = init_weights([8, 8, 32, 64])     # 3x3x32 conv, 64 outputs
w3 = init_weights([8, 8, 64, 128])    # 3x3x32 conv, 128 outputs
w4 = init_weights([8, 8, 64, 128])    # 3x3x32 conv, 128 outputs
w5 = init_weights([5, 5, 64, 256])    # 3x3x32 conv, 128 outputs
w6 = init_weights([256 * 5 * 5, 2048]) # FC 128 * 4 * 4 inputs, 625 outputs
wo = init_weights([2048, 2])         # FC 625 inputs, 10 outputs (labels)
p_keep_conv = tf.constant(.7)
p_keep_hidden = tf.constant(.5)
layers = Layers(w1, w2, w3, w4, wo, p_keep_conv, p_keep_hidden)


batch_size = 16

X = tf.placeholder( tf.float32, [None, 300, 300, 3] )
Y = tf.placeholder( tf.float32, [None, 2] )
py_x = layers.esimate(X)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.AdamOptimizer(0.001, 0.90).minimize(cost)
predict_op = tf.argmax(py_x, 1)
test_op = tf.argmax(Y,1)
test_accuracy = tf.reduce_mean( tf.cast( tf.equal(predict_op, test_op), tf.float32 ) )

#######################################

# Start a new session to show example output.
saver = tf.train.Saver()

with tf.Session() as sess:


    # Coordinate the loading of image files.
    tf.global_variables_initializer().run()


    for i in range(50):
        # Make Batch of Image
        np.random.shuffle(train_paths)

        for j in range( batch_size, len(train_paths), batch_size ):
            train_paths_part = train_paths[j-batch_size:j]

            train_images_part, train_labels_part = get_batch_image_label(train_paths_part)
            print('epoch',i,'iter:',j,':',sess.run( [test_accuracy,cost], feed_dict={X: train_images_part, Y:train_labels_part } ))

        save_path = saver.save(sess, "tmp/model.ckpt")
        print("Model saved in file: %s" % save_path)
