import numpy as np
import tensorflow as tf


def gram_matrix(x):
    batch_size, height, width, channels = x.shape
    features = np.reshape(x, (batch_size, -1, channels))
    features_T = np.transpose(features, [0, 2, 1])
    gram = np.matmul(features_T, features) / (height * width * channels)
    return gram


def vgg_preprocess(x, mean):
    x = x[..., ::-1]  # RGB -> BGR
    return tf.nn.bias_add(x, -mean)


def style_grams(style_image, style_layers=['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3']):
    grams_per_layer = {}
    with tf.Graph().as_default(), tf.device('/CPU:0'), tf.Session() as sess:
        imagenet_mean = tf.constant([103.939, 116.779, 123.68])
        input_image = tf.placeholder(dtype=tf.float32, shape=style_image.shape)
        preprocessed_image = vgg_preprocess(input_image, imagenet_mean)
        vgg = tf.keras.applications.VGG16(input_tensor=preprocessed_image, weights='imagenet', include_top=False)
        outputs_dict = dict([(layer.name, layer.output) for layer in vgg.layers])

        for layer_name in style_layers:
            features = sess.run(outputs_dict[layer_name], feed_dict={input_image: style_image})
            gram = gram_matrix(features)
            grams_per_layer[layer_name] = gram
    return grams_per_layer
