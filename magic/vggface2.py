import numpy as np
import time
import cv2

vf2root='J:/vggface2'

trainlist = vf2root+'/train_list.txt'
testlist = vf2root+'/test_list.txt'

train_imgs = vf2root+'/train/'
test_imgs = vf2root+'/test/'

def readall(fn):
    t = time.time()
    print('reading from', fn)
    with open(fn, 'r') as f:
        k = f.read()
    t = time.time()-t
    print(fn, 'read into memory ({:.2f}s)'.format(t))
    return k

class vggface2:
    def __init__(self, listfile, imgpath):
        self.listfile = listfile
        self.imgpath = imgpath

        def listread(fn):
            tl = trainlist = readall(fn)

            tl = tl.split('\n')

            d = {}
            l = []
            n = 0

            for line in tl:
                line = line.strip()
                if len(line)==0:
                    continue
                else:
                    identity = int(line[1:1+6])

                    fn = line

                    if identity in d:
                        d[identity].append(fn)
                    else:
                        n+=1
                        k = [fn]
                        l.append(k)
                        d[identity] = k

            # some identities lost their images (broken archive?)
            # for ll in l:
            #     p = self.imgpath + ll[0]
            #     import os
            #     if not os.path.exists(p):
            #         print(p, 'does not exist')

            return n,l,d

        self.n, self.l, self.d = listread(self.listfile)
        print(self.n,'identities')

    def randone(self, side=64):
        index = np.random.choice(self.n)
        imglist = self.l[index]
        imgpath = self.imgpath + np.random.choice(imglist)

        # return index, imgpath

        img = cv2.imread(imgpath)
        # if img is None:
            # print(imgpath)
        assert img is not None

        img = randomcrop(img, side)

        return index, img

    def minibatch(self, batch_size, side=64):
        mbid, mbim = list(zip(*[self.randone(side=side) for i in range(batch_size)]))

        mbid = np.array(mbid)
        mbim = np.array(mbim)

        return mbid, mbim

from cv2tools import vis

def randomcrop(img, side):
    # crop sidexside randomly from given image.
    h,w = img.shape[0:2]
    minside = min(h,w)

    scale = minside/side
    # assert scale > 1

    cs = chosen_scale = np.random.uniform(scale*.46, scale*.95)

    img = vis.resize_perfect(img, h/cs, w/cs, cubic=True)

    h,w = img.shape[0:2]

    sh,sw = np.random.choice(h-side), np.random.choice(w-side)

    cropped = img[sh:sh+side, sw:sw+side, :]

    if np.random.uniform()>0.5:
        cropped = np.flip(cropped, axis=1)

    return cropped

vf2t = vggface2(trainlist, train_imgs)
# vf2s = vggface2(testlist, test_imgs)

def test():
    # print(vf2t.minibatch(5))
    # for i in range(10):
    #     print(vf2t.randone())

    # for i in range(20):
    #     a,b = vf2t.randone()
    #     cv2.imshow('yolo', b)
    #     cv2.waitKey(0)

    lb, imgs = vf2t.minibatch(batch_size=20, side=64)
    print(lb, imgs.shape)

if __name__ == '__main__':
    test()
