import matplotlib.pyplot as plt
import tensorflow as tf

image_raw_data=tf.gfile.FastGFile('caffe01.jpg','rb').read()

with tf.Session() as sess:
    image_data=tf.image.decode_jpeg(image_raw_data)
    print(image_data.eval())
    print("...........")
    #plt.imshow(image_data.eval())
    #plt.show()

    image_data=tf.image.convert_image_dtype(image_data,dtype=tf.float32)
    #img_encode=tf.image.encode_jpeg(image_data)
    print(image_data.eval())
 #   with tf.gfile.GFile('caffe_convert.jpg','wb') as f:
  #      f.write(img_encode.eval())
    resize_img=tf.image.resize_images(image_data,(300,300),method=0)
    print resize_img.get_shape()
    croped=tf.image.resize_image_with_crop_or_pad(image_data,300,300)
    central_cropped=tf.image.central_crop(image_data,0.5)
    flipped=tf.image.flip_up_down(image_data)
    flipped=tf.image.flip_left_right(image_data)
    transposed=tf.image.transpose_image(image_data)
    flipped=tf.image.random_flip_up_down(image_data)
    flipped=tf.image.random_flip_left_right(image_data)
    adjusted=tf.image.adjust_brightness(image_data,-0.5)
    adjusted=tf.image.adjust_brightness(image_data,0.5)
    adjusted=tf.image.random_brightness(image_data,0.5)
    adjusted=tf.image.adjust_contrast(image_data,-5)
    adjusted=tf.image.random_contrast(image_data,0,5)
    adjusted=tf.image.adjust_hue(image_data,0.1)
    adjusted=tf.image.adjust_hue(image_data,0.3)
    adjusted=tf.image.adjust_hue(image_data,0.6)
    adjusted=tf.image.adjust_hue(image_data,0.0)
    adjusted=tf.image.random_hue(image_data,0.5)
    adjusted=tf.image.adjust_saturation(image_data,-5)
    adjusted=tf.image.adjust_saturation(image_data,5)
    adjusted=tf.image.random_saturation(image_data,0,5)
    adjusted=tf.image.per_image_whitening(image_data)
    plt.imshow(adjusted.eval())
    plt.show()
