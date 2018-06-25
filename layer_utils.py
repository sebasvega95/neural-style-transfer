import tensorflow as tf


def ConvBnAct(filters, kernel_size=(3, 3), stride=1, activation='relu', name=''):
    def forward(x):
        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            strides=stride,
            padding='same',
            name=name + '_conv')(x)
        x = tf.keras.layers.BatchNormalization(name=name + '_batchnorm')(x)
        x = tf.keras.layers.Activation(activation, name=name + '_' + activation)(x)
        return x

    return forward


def DeconvBnAct(filters, kernel_size=(3, 3), stride=1, activation='relu', name=''):
    def forward(x):
        x = tf.keras.layers.Conv2DTranspose(
            filters,
            kernel_size,
            strides=stride,
            padding='same',
            name=name + '_conv')(x)
        x = tf.keras.layers.BatchNormalization(name=name + '_batchnorm')(x)
        x = tf.keras.layers.Activation(activation, name=name + '_' + activation)(x)
        return x

    return forward


def ResidualBlock(filters, kernel_size=(3, 3), stride=1, first=False, name=''):
    def forward(x):
        x_shortcut = x

        # don't repeat bn->relu since we just did it
        if not first:
            x = tf.keras.layers.BatchNormalization(name=name + '_batchnorm1')(x)
            x = tf.keras.layers.Activation('relu', name=name + '_relu1')(x)
        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            strides=stride,
            padding='same',
            name=name + '_conv1')(x)

        x = tf.keras.layers.BatchNormalization(name=name + '_batchnorm2')(x)
        x = tf.keras.layers.Activation('relu', name=name + '_relu2')(x)
        x = tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            padding='same',
            name=name + '_conv2')(x)

        x = tf.keras.layers.Add(name=name + '_shortcut')([x, x_shortcut])
        return x

    return forward
