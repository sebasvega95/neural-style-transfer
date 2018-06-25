import tensorflow as tf

from layer_utils import ConvBnAct, DeconvBnAct, ResidualBlock


def normalize_image(x):
    return x / 255.0


def denormalize_image(x):
    return (x + 1) * 127.5


def create_resnet(input_shape, name='resnet'):
    with tf.variable_scope(name):
        x_input = tf.keras.layers.Input(input_shape, name='input')

        x = tf.keras.layers.Lambda(normalize_image, name='normalize')(x_input)
        x = ConvBnAct(32, (9, 9), name='conv1')(x)
        x = ConvBnAct(64, (3, 3), stride=2, name='conv2')(x)
        x = ConvBnAct(128, (3, 3), stride=2, name='conv3')(x)

        x = ResidualBlock(128, (3, 3), stride=1, name='resblock1', first=True)(x)
        x = ResidualBlock(128, (3, 3), stride=1, name='resblock2')(x)
        x = ResidualBlock(128, (3, 3), stride=1, name='resblock3')(x)
        x = ResidualBlock(128, (3, 3), stride=1, name='resblock4')(x)
        x = ResidualBlock(128, (3, 3), stride=1, name='resblock5')(x)

        x = DeconvBnAct(64, (3, 3), stride=2, name='deconv1')(x)
        x = DeconvBnAct(32, (3, 3), stride=2, name='deconv2')(x)
        x = DeconvBnAct(3, (9, 9), activation='tanh', name='deconv3')(x)
        x = tf.keras.layers.Lambda(denormalize_image, name='denormalize')(x)

        model = tf.keras.models.Model(inputs=x_input, outputs=x, name=name)
        return model
