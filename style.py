import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import tensorflow as tf

from model import create_resnet
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content', type=str, help='Path to content image')
    parser.add_argument('--gen', type=str, help='Path where to save the generated image')
    parser.add_argument('--weights', type=str, help='Path to model\'s weights', required=True)
    parser.add_argument('--webcam', action='store_true', help='Generate images from the webcam')
    return parser.parse_args()


def check_args(args):
    if args.gen is not None:
        if args.webcam:
            print('WARN: Ignoring webcam argument since you specified a content image')
        if not utils.path_exists(args.content):
            print('Content image not found in', args.content)
            exit(-1)
    elif not args.webcam:
        print('Please specify a content image with --content, or use --webcam to generate from your webcam')
    if not utils.path_exists(args.weights):
        print('Weights not found in', args.weights)
        exit(-1)
    if args.gen is not None:
        pathlib.Path(args.gen).parent.mkdir(parents=True, exist_ok=True)


args = parse_args()
check_args(args)

tf.reset_default_graph()
transformation_model = create_resnet(input_shape=(None, None, 3))
transformation_model.load_weights(args.weights)

if args.gen is not None:
    content_image = utils.preprocess_image_from_path(args.content)
    gen = transformation_model.predict(content_image)

    gen = np.squeeze(gen)
    gen = gen.astype(np.uint8)

    if args.gen is None:
        plt.figure()
        plt.imshow(gen)
        plt.axis('off')
        plt.show()
    else:
        gen = cv2.cvtColor(gen, cv2.COLOR_RGB2BGR)
        cv2.imwrite(args.gen, gen)
elif args.webcam:
    print('No content image specified, using webcam')
    cam_capture = cv2.VideoCapture(0)
    if not cam_capture.isOpened():
        print('No webcam found')
        exit(-1)
    print('Press q to quit')
    while True:
        _, frame = cam_capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        frame = np.expand_dims(frame, axis=0)

        gen = transformation_model.predict(frame)
        gen = np.squeeze(gen)
        gen = gen.astype(np.uint8)
        gen = cv2.cvtColor(gen, cv2.COLOR_RGB2BGR)
        cv2.imshow('Generated image', gen)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    cam_capture.release()
    cv2.destroyAllWindows()
