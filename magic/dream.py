import numpy as np
import cv2

from cv2tools import filt,vis
from canton import *

import tensorflow as tf

import sys
sys.path.append('..')
from image import load_image

from test import forenet, forenet_infer

def feed_gen(img):
    input_img_ph = ph([None,None,None])

    canvas = tf.Variable(
        tf.random_normal(shape=(1,)+img.shape, mean=0.5, stddev=.01)) # gray
    target = tf.Variable(tf.fill((1,)+img.shape, 0.5)) # gray

    assign_target = tf.assign(target, tf.cast(input_img_ph, tf.float32)/255.)

    targfeat = forenet(target)
    canvfeat = forenet(tf.random_normal(shape=tf.shape(canvas), stddev=.001)+canvas)
    canvfeat = forenet(canvas)

    loss = tf.reduce_mean(tf.square(targfeat-canvfeat)) * 10 + tf.reduce_mean(tf.square(tf.square(canvas-0.5))) * 10

    opt = tf.train.AdamOptimizer(
        learning_rate=0.1,
        beta1=0.9,
        beta2=0.999,
    )

    updateop = opt.minimize(loss, var_list = [canvas])


    def settarg(img):
        k = img.view()
        k.shape = (1,) +k.shape
        sess = get_session()
        sess.run([assign_target], feed_dict={input_img_ph:k})

    def update():
        sess = get_session()
        _, l = sess.run([updateop, loss])
        return l

    def getcanvas():
        sess = get_session()
        c = sess.run([canvas])[0]
        return c[0]

    return settarg, update, getcanvas

jeff = load_image('../jeff.jpg')

h,w = jeff.shape[0:2]
jeff = vis.resize_perfect(jeff, 192, 192)
print(jeff.shape)

settarg, update, getcanvas = feed_gen(jeff)

get_session().run(gvi())
settarg(jeff)

def show():
    vis.show_autoscaled(jeff,name='jeff')
    canvas = getcanvas()
    vis.show_autoscaled(canvas)
    cv2.waitKey(1)

show()

def r(ep=100):
    for i in range(ep):
        loss = update()

        if i % 10 == 0:
            print('ep({}/{})'.format(i+1, ep))
            print('loss {:2.8f}'.format(loss))

        if (i+1) % 50 == 0:
            show()
