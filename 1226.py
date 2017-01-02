import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import tensorflow as tf

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w1, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w1,                       # l1a shape=(?, 28, 28, 32)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],              # l1 shape=(?, 14, 14, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,                     # l2a shape=(?, 14, 14, 64)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],              # l2 shape=(?, 7, 7, 64)
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,                     # l3a shape=(?, 7, 7, 128)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],              # l3 shape=(?, 4, 4, 128)
                        strides=[1, 2, 2, 1], padding='SAME')

    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])    # reshape to (?, 2048)
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
    return pyx


def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_jpeg(file_contents, channels=3)
    return example, label



################################################################
# Get List of TEST of TRAIN Paths

train_paths = []
train_labels = []
test_paths = []
test_labels = []

TRAIN_DIR = 'input/train/'
TEST_DIR = 'input/test/'
train_imnames = os.listdir(TRAIN_DIR)
test_imnames = os.listdir(TEST_DIR)

for train_imname in train_imnames:
    train_paths.append(os.path.join(TRAIN_DIR, train_imname))
    cat_or_dog = train_imname.split('.')[0]

    if cat_or_dog == 'cat':
        train_labels.append([1, 0])
    else:
        train_labels.append([0, 1])
for test_imname in train_imnames:
    test_paths.append(os.path.join(TEST_DIR, test_imname))
    cat_or_dog = test_imname.split('.')[0]

    if cat_or_dog == 'cat':
        test_labels.append(0)
    else:
        test_labels.append(1)

w1 = init_weights([7,7,3,32])
w2 = init_weights([5,5,32,48])
w3 = init_weights([3,3,48,64])
w4 = init_weights([64, 625])
wo = init_weights([625, 2])

print(w1, w2, w3, w4, wo)

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

#X = tf.placeholder( "float", [None, 500, 500, 3] )
#Y = tf.placeholder( "float", [None, 2] )

#######################################
# Makes an input queue
# Make a queue of file names including all the JPEG images files in the relative
# image directory.
filename_queue = tf.train.slice_input_producer([train_paths,train_labels])

# Read a whole file from the queue, the first returned value in the tuple is the
# filename which we are ignoring.
image_file = tf.read_file(filename_queue[0])

# Decode the image as a JPEG file, this will turn it into a Tensor which we can
# then use in training.
image = tf.image.decode_jpeg(image_file, channels=3)
image = tf.image.resize_image_with_crop_or_pad(image,500,500)
image = tf.cast( image, tf.float32 )

label = filename_queue[1]

# Start a new session to show example output.
with tf.Session() as sess:
    # Required to get the filename matching to run.
    tf.initialize_all_variables().run()


    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    batch_size = 64
    epoch_size = 10

    # Make Batch of Image
    image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                      batch_size=batch_size,
                                                      capacity=5000,
                                                      min_after_dequeue=1000
                                                      )
    X = image_batch
    Y = label_batch

    py_x = model(X, w1, w2, w3, w4, wo, p_keep_conv, p_keep_hidden)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
    train_op = tf.train.AdamOptimizer(0.001, 0.9).minimize(cost)
    predict_op = tf.argmax(py_x, 1)

    for i in range(epoch_size):
        sess.run(py_x)

    # Finish off the filename queue coordinator.
    coord.request_stop()
    coord.join(threads)
