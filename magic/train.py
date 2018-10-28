from canton import *
import tensorflow as tf
from vggface2 import vf2t

def net():

    c = Can()
    c.add(Conv2D(3,16, k=3, std=1, stddev=2))
    c.add(Act('relu'))
    c.add(Conv2D(16,16, k=3, std=1, stddev=2))
    c.add(Act('relu'))
    c.add(Conv2D(16,32, k=3, std=2, stddev=2))
    c.add(Act('relu'))
    c.add(Conv2D(32,32, k=3, std=2, stddev=2))
    c.add(Act('relu'))
    c.add(Conv2D(32,32, k=3, std=2, stddev=2))
    c.add(Act('relu'))
    c.add(Conv2D(32,32, k=3, std=2, stddev=2))
    c.add(Act('relu'))
    c.add(Conv2D(32,32, k=3, std=2, stddev=2))
    c.add(Act('relu'))
    c.add(Conv2D(32,32, k=3, std=1, stddev=2))

    c.add(Lambda(lambda x:tf.reduce_mean(x, axis=[1,2])))
    c.add(Dense(32, vf2t.n, bias=False))

    c.chain()

    return c

net = net()
net.summary()

def feed_gen():

    labeld, imgd = tf.placeholder(tf.int32), ph([None,None,None])

    imgp = imgd/255.

    pred = net(imgd)

    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labeld, logits=pred)
        )

    acc = class_accuracy(pred, labeld)

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

def r(ep=100):
    for i in range(ep):
        print('ep({}/{})'.format(i+1, ep))
        loss, acc = feed(vf2t.minibatch(batch_size=128, side=64))
        print('loss {:2.4f} acc {:2.5f}'.format(loss, acc))
