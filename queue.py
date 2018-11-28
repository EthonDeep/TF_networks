import tensorflow as tf

queue=tf.FIFOQueue(2,"int32")
init=queue.enqueue_many(([0,10],))
x=queue.dequeue()

y=x+1

q_in=queue.enqueue([y])

with tf.Session() as sess:
    init.run()
    for _ in range(5):
        v,_=sess.run([x,q_in])
        print v
