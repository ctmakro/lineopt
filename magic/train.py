from canton import *
import tensorflow as tf
from vggface2 import getvf2t,getvf2s

vf2t = getvf2s()

def net():

    c = Can()
    c.add(Conv2D(3,16, k=3, std=1, stddev=2))
    c.add(Act('relu'))
    c.add(Conv2D(16,32, k=3, std=1, stddev=2))
    c.add(Act('relu'))
    c.add(Conv2D(32,32, k=5, std=2, stddev=2))
    c.add(Act('relu'))
    c.add(Conv2D(32,32, k=5, std=2, stddev=2))
    c.add(Act('relu'))
    c.add(Conv2D(32,64, k=5, std=2, stddev=2))
    c.add(Act('relu'))
    c.add(Conv2D(64,64, k=5, std=2, stddev=2))
    c.add(Act('relu'))
    c.add(Conv2D(64,64, k=5, std=2, stddev=2))
    c.add(Act('relu'))
    c.add(Conv2D(64,64, k=3, std=1, stddev=2))

    c.add(Lambda(lambda x:tf.reduce_mean(x, axis=[1,2])))
    c.add(Dense(64, vf2t.n, bias=True))

    c.chain()

    return c

net = net()
net.summary()

def feed_gen():

    labeld, imgd = tf.placeholder(tf.int32), ph([None,None,None])

    imgp = imgd/255.

    pred = net(imgp)

    labelh = tf.one_hot(labeld, vf2t.n)
    
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labeld, logits=pred)
        )
    loss = mean_softmax_cross_entropy(pred, labelh)

    acc = class_accuracy(pred, labeld)
    acc = one_hot_accuracy(pred, labelh)

    opt = tf.train.AdamOptimizer(
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
    )

    train_step = opt.minimize(loss, var_list=net.get_weights())

    def feed(t):
        label, imgs = t
        sess = get_session()
        _,l,a = sess.run([train_step,loss,acc], feed_dict={
            labeld:label, imgd:imgs
        })
        return l,a

    return feed

feed = feed_gen()
get_session().run(gvi())

from llll import MultithreadedGenerator as MG

mg = MG(lambda:vf2t.minibatch(batch_size=64, side=64), 320, ncpu=40)

def r(ep=100):
    for i in range(ep):
        print('ep({}/{})'.format(i+1, ep))
        loss, acc = feed(mg.get())
        print('loss {:2.4f} acc {:2.5f}'.format(loss, acc))
