import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def residual_block(x, filters, kernel_size=(3, 3), stride=1, conv_shortcut=False):
    shortcut = x

    # Main path
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, kernel_size, padding='same',
                      kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)

    # Shortcut path
    if conv_shortcut:
        shortcut = layers.Conv2D(filters, (1, 1), strides=stride, padding='same',
                                 kernel_regularizer=regularizers.l2(1e-4))(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([shortcut, x])
    x = layers.ReLU()(x)
    return x

def ResNet18(input_shape=(512, 512, 6), num_classes=1):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same',
                      kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = residual_block(x, 64)
    x = residual_block(x, 64)

    x = residual_block(x, 128, stride=2, conv_shortcut=True)
    x = residual_block(x, 128)

    x = residual_block(x, 256, stride=2, conv_shortcut=True)
    x = residual_block(x, 256)

    x = residual_block(x, 512, stride=2, conv_shortcut=True)
    x = residual_block(x, 512)

    # Decoder / Upsampling path
    x = layers.Conv2DTranspose(256, 3, strides=2, padding='same')(x)  # 16x16 -> 32x32
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(128, 3, strides=2, padding='same')(x)  # 32x32 -> 64x64
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same')(x)   # 64x64 -> 128x128
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(32, 3, strides=2, padding='same')(x)   # 128x128 -> 256x256
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(16, 3, strides=2, padding='same')(x)   # 256x256 -> 512x512
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Final segmentation mask (same size as input)
    outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    return model
