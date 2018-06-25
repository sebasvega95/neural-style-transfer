import argparse
import os
import pathlib
import shutil
import tensorflow as tf

from loss import get_total_loss
from model import create_resnet
import precompute
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--style', type=str, help='Path to style image', required=True)
    parser.add_argument('--train', type=str, help='Path to training (content) images', required=True)
    parser.add_argument('--weights', type=str, help='Path where to save the model\'s weights', required=True)
    return parser.parse_args()


def check_args(args):
    if not utils.path_exists(args.style):
        print('Style image not found in', args.style)
        exit(-1)
    if not utils.path_exists(args.train):
        print('Train path', args.style, 'not found')
        exit(-1)
    pathlib.Path(args.weights).parent.mkdir(parents=True, exist_ok=True)


args = parse_args()
check_args(args)

epochs = 2
batch_size = 4
img_height = 256
img_width = 256

num_images = utils.get_num_images(args.train)
if num_images <= 0:
    print('No images found in', args.train)

tf.reset_default_graph()

style_image = utils.preprocess_image_from_path(args.style)
print('Precomputing style gram matrices')
style_grams = precompute.style_grams(style_image)

print('Creating model')
transformation_model = create_resnet(input_shape=(None, None, 3), name='transformation_net')

with tf.Session() as sess:
    print('Defining loss')
    total_loss = get_total_loss(
        transformation_model,
        style_grams,
        content_weight=2,
        style_weight=1e2,
        total_variation_weight=1e-5,
        batch_size=batch_size,
        name='loss')

    with tf.name_scope('optimizer'):
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='transformation_net')
        extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        optimizer = tf.train.AdamOptimizer()
        with tf.control_dependencies(extra_ops):
            train_op = optimizer.minimize(total_loss, var_list=var_list)

    opt_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='optimizer')
    model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='transformation_net')
    init = tf.variables_initializer(var_list=opt_vars + model_vars)
    sess.run(init)

    print('Training')
    shutil.rmtree('logs', ignore_errors=True)
    summaries = tf.summary.merge_all()
    img_gen = utils.image_generator(args.train, batch_size=batch_size, target_shape=(img_width, img_height))
    writer = tf.summary.FileWriter('logs', sess.graph)
    global_step = 0
    for epoch in range(epochs):
        step = 0
        while step * batch_size < num_images:
            images = next(img_gen)
            _, step_loss, summary = sess.run(
                [train_op, total_loss, summaries],
                feed_dict={
                    transformation_model.input: images,
                    tf.keras.backend.learning_phase(): 1
                })
            if step % 500 == 0:
                print('Epoch {}, step {}:   loss: {}'.format(epoch, step, step_loss))
                writer.add_summary(summary, global_step)
            step += 1
            global_step += 1
    transformation_model.save_weights(args.weights)
