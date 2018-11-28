import tensorflow as tf
from scipy.misc import imsave
import matplotlib.pyplot as plt
reader=tf.TFRecordReader()
filename_queue=tf.train.string_input_producer(["./cifar10_test.tfrecords"])

_,serialized_example=reader.read(filename_queue)

features=tf.parse_single_example(
       serialized_example,
       features={
              'image_raw':tf.FixedLenFeature([],tf.string),
              'label':tf.FixedLenFeature([],tf.int64)
                })
images=tf.decode_raw(features['image_raw'],tf.uint8)
images=tf.reshape(images,(32,32,3))
labels=tf.cast(features['label'],tf.int32)

sess=tf.Session()

coord=tf.train.Coordinator()
threads=tf.train.start_queue_runners(sess=sess,coord=coord)

for i in range(10):
    image,label=sess.run([images,labels])
    print label
    imsave(str(label)+'.jpg',image)
    plt.imshow(image)
    plt.show()

sess.close()
coord.request_stop()
coord.join(threads)
