import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy import misc
import os
import tensorflow as tf


# Create some variables.
v1 = tf.Variable(1, name="w1")

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, "tmp/model.ckpt")
    print("Model restored.")