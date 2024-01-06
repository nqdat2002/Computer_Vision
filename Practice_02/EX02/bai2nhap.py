import tensorflow as tf
import tensorflow_datasets as tfds

# Construct a tf.data.Dataset
ds = tfds.load('open_images/v7', split='train', shuffle_files=True)

