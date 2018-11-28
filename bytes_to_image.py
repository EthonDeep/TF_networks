import numpy as np
from scipy.misc import imsave
def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

cifar10_file_dir='./cifar-10-batches-py/'


for i in range(1,6):
     batch_name=cifar10_file_dir+"data_batch_"+str(i)
     bytes_data=unpickle(batch_name)
     print batch_name +'on loading......'

     for j in range(0,10000):
          img=np.reshape(bytes_data['data'][j],(3,32,32))
          img=img.transpose(1,2,0)
          save_name='train/'+str(bytes_data['labels'][j])+'_'+str(i+(j-1)*10000)+'.jpg'
          imsave(save_name,img)
     print batch_name +'image trans finished!'

