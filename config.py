import tensorflow as tf
from utils.acc_patch import miou_binary
import math

## ---- super-parameter for model training ---- ##
patch_size = 512
patch_overlap = 512
num_bands = 6
epochs = 100
lr = 0.002
batch_size = 8
buffer_size = 200
size_scene = 102
step_per_epoch = math.ceil(size_scene/batch_size)

def dice_loss():
    def loss(y_true, y_pred, smooth = 1e-6):
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        
        intersection = tf.reduce_sum(tf.abs(y_true * y_pred), axis=1)
        dice = (2. * intersection) / (tf.reduce_sum(tf.square(y_true), axis=1) + tf.reduce_sum(tf.square(y_pred), axis=1) + smooth)
        
        loss = tf.reduce_mean(1 - dice)
        return loss
    return loss

def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        log_pt = tf.math.log(pt)

        focal_weight = tf.where(
            tf.equal(y_true, 1),
            alpha * tf.pow(1 - pt, gamma),
            (1 - alpha) * tf.pow(pt, gamma)
        )

        loss = -focal_weight * log_pt
        return tf.reduce_mean(loss)
    
    return loss

# Binary cross entropy with clip
def binary_crossentropy():
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)  # prevent log(0)
        loss = -tf.reduce_mean(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        return loss
    return loss

def focal_dice_loss():
    def loss(y_true, y_pred):
        loss = focal_loss()(y_true, y_pred) + dice_loss()(y_true, y_pred)
        return loss
    return loss

def bce_dice_loss():
    def loss(y_true, y_pred):
            
        dice = dice_loss()(y_true, y_pred)
            
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)  # prevent log(0)

        bce = - tf.reduce_mean(y_true * tf.math.log(y_pred) + 
                           (1 - y_true) * tf.math.log(1 - y_pred))
        
        loss = bce + dice
        return loss
    return loss

## ---- configuration for model training ---- ##
class lr_schedule(tf.keras.optimizers.schedules.LearningRateSchedule):    
  def __init__(self, initial_learning_rate, steps_all):
      self.initial_learning_rate = initial_learning_rate
      self.steps_all = steps_all
  def __call__(self, step):
     return self.initial_learning_rate*((1-step/self.steps_all)**0.9)

# Loss functions
loss_dice = dice_loss()
loss_focal = focal_loss()
loss_focal_dice = focal_dice_loss()
loss_bce_dice = bce_dice_loss()
loss_bce = binary_crossentropy() 

# Optimizer
opt_adam = tf.keras.optimizers.Adam(learning_rate=lr_schedule(lr,step_per_epoch*epochs))

## ---- metrics ---- ##
tra_loss = tf.keras.metrics.Mean(name="tra_loss")
tra_oa = tf.keras.metrics.BinaryAccuracy('tra_oa')
tra_miou = miou_binary(num_classes=2, name='tra_miou')
test_loss = tf.keras.metrics.Mean(name="test_loss")
test_oa = tf.keras.metrics.BinaryAccuracy('test_oa')
test_miou = miou_binary(num_classes=2, name='test_miou')


