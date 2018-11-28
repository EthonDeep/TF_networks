import tensorflow as tf
import numpy as np
from scipy.misc import imsave
import matplotlib.pyplot as plt 
reader=tf.TFRecordReader()
filename_queue=tf.train.string_input_producer(["./cifar10_train.tfrecords"])

_,serialized_example=reader.read(filename_queue)

features=tf.parse_single_example(
serialized_example,
features={
      'image_raw':tf.FixedLenFeature([],tf.string),
      'label':tf.FixedLenFeature([],tf.int64)
        })
images=tf.decode_raw(features['image_raw'],tf.uint8)
images.set_shape([32*32*3])
#images=tf.reshape(images,[32,32,3])
labels=tf.cast(features['label'],tf.int32)
labels = tf.one_hot(labels, depth=10)

min_after_queue=1000
batch_size=100
capacity=min_after_queue+3*batch_size
image_batch,label_batch=tf.train.shuffle_batch([images,labels],batch_size,capacity,min_after_queue)




X=tf.placeholder(tf.float32,[None,32*32*3])
Y_=tf.placeholder(tf.float32,[None,10])

w=tf.Variable(tf.random_normal([32*32*3,10],stddev=0.1))
b=tf.Variable(tf.zeros([10]))

y=tf.matmul(X,w)+b


count=0

cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=Y_))

train_op=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
correct_prediction = tf.equal(tf.nn.softmax(y),Y_)
acc=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
sess=tf.Session()

sess.run(tf.global_variables_initializer())
coord=tf.train.Coordinator()
threads=tf.train.start_queue_runners(sess,coord=coord)
for step in range(1):
    for i in range(100):
       image_array,label_array=sess.run([image_batch,label_batch])
       
       #print image_array.shape
       #label_array=np.reshape(label_array,(1,10))
       #print label_array
       #print sess.run(tf.argmax(label_array.reshape(10)))
       #print sess.run(Y_,feed_dict={X:image_array,Y_:label_array})
       #print sess.run(tf.nn.softmax(y),feed_dict={X:image_array,Y_:label_array})
       print('........')
       acc1=sess.run(acc,feed_dict={X:image_array,Y_:label_array})
       print str(acc1*100)+'%'
#       imsave(str(label_array)+'.jpg',image_array)
       #plt.imshow(image_array.reshape([32,32,3]))
       #plt.show()
#       sess.run([train_op],feed_dict={X:image_array,Y_:label_array})
  # print('training acc %s'% acc)



sess.close()
coord.request_stop()
coord.join(threads)
