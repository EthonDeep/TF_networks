import tensorflow as tf
import numpy as np
import scipy.misc as smc
import os
import matplotlib.pyplot as plt
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

img_dir='./test/'
sess=tf.Session()
tfrecords_name="./cifar10_test.tfrecords"
writer=tf.python_io.TFRecordWriter(tfrecords_name)
for filename in os.listdir(img_dir):
    #print filename
    split_name=filename.split('_')
    label_name=split_name[0]
    print 'processing image... '
    image_raw=smc.imread(img_dir+filename).tostring()
    example=tf.train.Example(features=tf.train.Features(feature={
              'image_raw':_bytes_feature(image_raw),
              'label':_int64_feature(int(label_name)) 
              }))
    writer.write(example.SerializeToString())

writer.close()
   # plt.imshow(image_raw)
   # plt.show()
sess.close() 
