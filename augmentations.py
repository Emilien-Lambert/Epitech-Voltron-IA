import cv2;
import tensorflow as tf

def blur(img):
    return (cv2.blur(img,(30,30)))

def horizontal_flip(img):
    return (tf.image.flip_left_right(img))

def vertical_flip(img):
    return (tf.image.flip_up_down(img))
 
def contrast(img):
    return (tf.image.adjust_contrast(img, 0.5))

def saturation(img):
    return (tf.image.adjust_saturation(img, 3))

def hue(img):
    return (tf.image.adjust_hue(img, 0.1)) 

def gamma(img):
    return (tf.image.adjust_gamma(img, 2))
