from canton import *
import tensorflow as tf
from vggface2 import getvf2t,getvf2s

def netg(nclass,feat=False):

    c = Can()

    # c.add(Conv2D(3,64, k=7, std=1, ))
    # rect()

    def conv(i,o,k=3):
        c.add(Conv2D(i,o,k=k,std=1,stddev=1))

    def b(k):
        c.add(BatchNorm(k))

    def rect():
        c.add(Act('tanh'))

    def ap():
        # c.add(MaxPool2D(k=2,std=2))
        c.add(AvgPool2D(k=2,std=2))

    conv(3,16);rect() #128
    conv(16,16);b(16);rect() #128
    ap() #64

    conv(16,32,k=5);b(32);rect() #64
    ap() #32

    conv(32,64);b(64);rect() #32

    conv(64,64);b(64);rect() #32
    ap() #16
    #--14

    conv(64,128,k=5);b(128);rect() #16
    ap()
    conv(128,128,k=5);b(128);rect() #8
    ap()
    # 25 above
    conv(128,128);b(128);rect() #4
    conv(128,64);b(64);rect() #4

    c.add(Lambda(lambda x:tf.reduce_mean(x, axis=[1,2])))

    c.chain()
    featnet = c

    c = Can()
    c.add(Dense(64, nclass, bias=True))
    c.chain()

    classnet = c

    c = Can()
    for i in featnet.subcans: c.add(i)
    for i in classnet.subcans: c.add(i)
    c.chain()

    net = c

    if feat==False:
        return c
    else:
        return featnet, classnet, net

def feed_gen():

    labeld, imgd = tf.placeholder(tf.int32), ph([None,None,None])

    imgp = imgd/255.

    feat = featnet(imgp)
    pred = classnet(feat)
    # labelh = tf.one_hot(labeld, vf2t.n)

    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labeld, logits=pred)
        )
    # loss = mean_softmax_cross_entropy(pred, labelh)

    # intra-class variance reduction
    bs = tf.shape(feat)[0]
    feat0=feat[0:bs//2]
    feat1=feat[bs//2:bs]

    featloss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(feat0-feat1), axis=1)))
    featloss = 1 - tf.reduce_mean(
        tf.reduce_sum(feat0*feat1, axis=1)/tf.sqrt(tf.reduce_sum(feat0*feat0,axis=1)*tf.reduce_sum(feat1*feat1,axis=1))
    )

    acc = class_accuracy(pred, labeld)
    # acc = one_hot_accuracy(pred, labelh)

    opt = tf.train.AdamOptimizer(
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
    )

    train_step = opt.minimize(loss+featloss*0.05, var_list=net.get_weights())

    def feed(t):
        label, imgs = t
        sess = get_session()
        _,l,fl,a = sess.run([train_step,loss,featloss,acc], feed_dict={
            labeld:label, imgd:imgs
        })
        return l,fl,a

    return feed

if __name__ == '__main__':

    vf2t = getvf2t()
    featnet, classnet, net = netg(vf2t.n, feat=True)
    net.summary()

    feed = feed_gen()
    get_session().run(gvi())

    from llll import MultithreadedGenerator as MG
    from llll import PoolMaster

    slave_code = '''
from vggface2 import getvf2t,getvf2s
vf2t = getvf2t()

mb = lambda bs:vf2t.minibatch(batch_size=bs, side=128)

from llll import PoolSlave
PoolSlave(mb)
    '''

    # mg = MG(lambda:vf2t.minibatch(batch_size=128, side=64), 320, ncpu=32)

    pm = PoolMaster(slave_code, 14)
    mg = MG(lambda:pm.call(32), 100, ncpu=20)

    def r(ep=100):
        for i in range(ep):
            print('ep({}/{})'.format(i+1, ep))
            loss, featloss, acc = feed(mg.get())
            print('loss {:2.4f} fl {:2.6f} acc {:2.5f}'.format(loss,featloss, acc))
