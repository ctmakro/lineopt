import cv2
import numpy as np
from cv2tools import vis,filt

def load_image(path, is_filename=True):
    if is_filename:
        # read image from file
        orig = cv2.imread(path)
        assert orig is not None
    else:
        # use in memory image
        orig = path
    return orig

def resize(image, maxwidth):
    cmw = max(image.shape[0:2])
    return vis.resize_perfect(image, maxwidth, maxwidth, cubic=True, a=3)

def artistic_enhance(image):
    # try penciling

    blurred = cv2.GaussianBlur(image,ksize=(0,0),sigmaX=20)

    alpha = 2.0
    beta = 0.7

    hf = (image - blurred)*alpha+beta
    return np.clip(hf, 0, 1)

def lru_cache(f):
    d = {}
    l = []

    def hash_args(*a, **kw):
        hash = '.'.join([str(id(i)) for i in a])
        hash += '|'
        hash += '?'.join([k+'%'+str(id(v)) for k,v in kw.items()])
        return hash

    def newf(*a, **kw):
        hash = hash_args(*a, **kw)
        if hash not in d:
            d[hash] = f(*a,**kw)

            l.append(hash)
            if len(l)>50:
                oldhash = l.popleft()
                d.pop(oldhash, None)

        return d[hash]
    return newf

@lru_cache
def fib(n):
    if n<=2:
        return 1
    else:
        return fib(n-1)+fib(n-2)

print(fib(50))

@lru_cache
def laplacian_pyramid(img, levels):
    gaussian_after_downsize = [img]
    laplacian = []

    for i in range(levels):
        orig = gaussian_after_downsize[-1]
        blurred = cv2.GaussianBlur(
            orig,
            ksize=(0,0),
            sigmaX=1.0,
        )

        laplacian.append(orig - blurred)

        downsized = vis.resize_nearest(
            blurred,
            blurred.shape[0]//2, blurred.shape[1]//2)

        gaussian_after_downsize.append(downsized)

    laplacian.append(blurred)
    return laplacian

if __name__ == '__main__':
    im = load_image('jeff.jpg')
    im = np.divide(im, 255., dtype='float32')
    lp = laplacian_pyramid(im, levels=4)
    for id,i in enumerate(lp):
        cv2.imshow(str(id), i)
        print(id, i.shape, i.dtype, i.max(), i.min())
    cv2.waitKey(0)
