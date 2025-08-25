'''Res-U-Net model as described by Zhang et al., 2018'''
import tensorflow as tf
from tensorflow.keras import layers, Model

def conv_block(x, filters):
    shortcut = x
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    # Residual connection
    shortcut = layers.Conv2D(filters, 1, padding="same")(shortcut)
    shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    return x

def encoder_block(x, filters):
    x = conv_block(x, filters)
    p = layers.MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(x, skip, filters):
    x = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding="same")(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x

def build_resunet(input_shape=(256, 256, 3), num_classes=1):
    inputs = layers.Input(shape=input_shape)

    # ---- Encoder ----
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    # ---- Bottleneck ----
    b1 = conv_block(p4, 1024)

    # ---- Decoder ----
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    # ---- Output ----
    if num_classes == 1:
        activation = 'sigmoid'  # binary segmentation
    else:
        activation = 'softmax'  # multi-class

    outputs = layers.Conv2D(num_classes, 1, padding='same', activation=activation)(d4)

    return Model(inputs, outputs, name="ResUNet")
