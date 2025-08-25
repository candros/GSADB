import tensorflow as tf
from tensorflow.keras import layers, models

# Basic residual block for ResNet-34
def residual_block(x, filters, stride=1, downsample=False, name=None):
    shortcut = x
    x = layers.BatchNormalization(name=name+'_bn0')(x)
    x = layers.ReLU(name=name+'_relu0')(x)
    
    if downsample:
        shortcut = x 
        
    x = layers.Conv2D(filters, kernel_size=3, strides=stride, padding='same', use_bias=False, name=name+'_conv1')(x)
    x = layers.BatchNormalization(name=name+'_bn1')(x)
    x = layers.ReLU(name=name+'_relu1')(x)
    
    x = layers.Conv2D(filters, kernel_size=3, strides=1, padding='same', use_bias=False, name=name+'_conv2')(x)
    #x = layers.BatchNormalization(name=name+'_bn2')(x)

    if downsample:
        shortcut = layers.Conv2D(filters, kernel_size=3, strides=stride, padding = 'same', use_bias=False, name=name+'_shortcut_conv')(shortcut)
        #shortcut = layers.BatchNormalization(name=name+'_shortcut_bn')(shortcut)

    x = layers.Add(name=name+'_add')([x, shortcut])
    x = layers.ReLU(name=name+'_relu2')(x)
    return x

# ResNet-34 encoder
def build_resnet34_encoder(input_tensor):
    x = layers.BatchNormalization()(input_tensor)
    x = layers.Conv2D(32, 7, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    skip1 = x # 256 x 256 x 32

    x = layers.MaxPooling2D(2, strides=1, padding='same')(x)

    # Layer 1
    x = residual_block(x, 64, stride = 2, downsample = True, name='conv2_block1') 
    for i in range(1,3):
        x = residual_block(x, 64, name=f'conv2_block{i+1}')

    skip2 = x  # 128 x 128 x 64

    # Layer 2
    x = residual_block(x, 128, stride=2, downsample=True, name='conv3_block1')
    for i in range(1, 4):
        x = residual_block(x, 128, name=f'conv3_block{i+1}')

    skip3 = x  # 64 x 64 x 128

    # Layer 3
    x = residual_block(x, 256, stride=2, downsample=True, name='conv4_block1')
    for i in range(1, 5):
        x = residual_block(x, 256, name=f'conv4_block{i+1}')

    skip4 = x  # 32 x 32 x 256

    # Layer 4
    x = residual_block(x, 512, stride=2, downsample=True, name='conv5_block1')
    for i in range(1, 3):
        x = residual_block(x, 512, name=f'conv5_block{i+1}')
    
    skip5 = x  # 16 x 16 x 512

    return skip1, skip2, skip3, skip4, skip5

# Decoder block
def upsample_block(x, filters, name,  skip = None, cat = True):
    x = layers.UpSampling2D((2, 2))(x)
    
    if cat:
        x = layers.Concatenate()([x, skip])
        
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

# Full ResUNet-34 model
def ResUNet34(input_shape=(512, 512, 6), num_classes=1):
    inputs = tf.keras.Input(shape=input_shape)

    skip1, skip2, skip3, skip4, skip5 = build_resnet34_encoder(inputs) # (256, 32), (128, 64), (64, 128), (32, 256), (16, 512)

    
    #bridge = layers.Conv2D(256, 3, padding='same')(skip4)
    bridge = layers.BatchNormalization(name='bridge')(skip5)
    bridge = layers.ReLU()(bridge) # bridge = 16 x 16 x 512

    # Concatenate blocks
    d1 = upsample_block(bridge, 256, 'decoder1', skip4) # d1 = 32 x 32 x 256
    d2 = upsample_block(d1, 128, 'decoder2', skip3) # d2 = 64 x 64 x 128
    d3 = upsample_block(d2, 64, 'decoder3', skip2) # d3 = 128 x 128 x 64
    d4 = upsample_block(d3, 32, 'decoder4', skip1) # d4 = 256 x 256 x 32
    d5 = upsample_block(d4, 16, 'decoder5', cat = False) # d5 = 512 x 512 x 16

    d6 = layers.Conv2D(input_shape[2], 3, padding='same')(d5)

    outputs = layers.Conv2D(num_classes, 1, activation='sigmoid')(d6)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

