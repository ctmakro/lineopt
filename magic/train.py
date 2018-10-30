from canton import *
import tensorflow as tf
from vggface2 import getvf2t,getvf2s

def netg(nclass):

    c = Can()

    # c.add(Conv2D(3,64, k=7, std=1, ))
    # rect()

    def b(k):
        c.add(BatchNorm(k))

    def rect():
        c.add(Act('tanh'))

    c.add(Conv2D(3,16, k=3, std=1, ))
    rect()

    c.add(Conv2D(16,32, k=3, std=1, ))
    b(32)
    rect()

    c.add(Conv2D(32,32, k=5, std=2, ))
    b(32)
    rect()

    c.add(Conv2D(32,32, k=5, std=2, ))
    b(32)
    rect()
    # 11 above
    c.add(Conv2D(32,64, k=5, std=2, ))
    b(64)
    rect()

    c.add(Conv2D(64,64, k=5, std=2, ))
    b(64)
    rect()

    c.add(Conv2D(64,64, k=5, std=2, ))
    b(64)
    rect()

    c.add(Conv2D(64,64, k=3, std=1, ))
    b(64)
    rect()

    c.add(Lambda(lambda x:tf.reduce_mean(x, axis=[1,2])))
    c.add(Dense(64, nclass, bias=True))

    c.chain()

    return c

def feed_gen():

    labeld, imgd = tf.placeholder(tf.int32), ph([None,None,None])

    imgp = imgd/255.

    pred = net(imgp)

    # labelh = tf.one_hot(labeld, vf2t.n)

    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labeld, logits=pred)
        )
    # loss = mean_softmax_cross_entropy(pred, labelh)

    acc = class_accuracy(pred, labeld)
    # acc = one_hot_accuracy(pred, labelh)

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

if __name__ == '__main__':

    vf2t = getvf2t()
    net = netg(vf2t.n)
    net.summary()

    feed = feed_gen()
    get_session().run(gvi())

    from llll import MultithreadedGenerator as MG
    from llll import PoolMaster

    slave_code = '''
from vggface2 import getvf2t,getvf2s
vf2t = getvf2t()

mb = lambda bs:vf2t.minibatch(batch_size=bs, side=64)

from llll import PoolSlave
PoolSlave(mb)
    '''

    # mg = MG(lambda:vf2t.minibatch(batch_size=128, side=64), 320, ncpu=32)

    pm = PoolMaster(slave_code, 14)
    mg = MG(lambda:pm.call(64), 100, ncpu=24)

    def r(ep=100):
        for i in range(ep):
            print('ep({}/{})'.format(i+1, ep))
            loss, acc = feed(mg.get())
            print('loss {:2.4f} acc {:2.5f}'.format(loss, acc))
