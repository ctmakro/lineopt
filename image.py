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

def f32(k):return np.array(k, dtype=np.float32)

# convert img type from uint8 to float32 properly
def zeroone(img):
    if img.dtype == 'uint8':
        # good luck if you succeeded with uint8
        img = np.divide(img, 255., dtype='float32')
    return img

def artistic_enhance(img):
    img = zeroone(img)

    blurred = cv2.GaussianBlur(img,ksize=(0,0),sigmaX=20)

    theta = f32(.3)
    alpha = f32(.7)

    mean = img.mean()

    hf = img - blurred
    res = (img + hf * theta) * alpha + (0.85 - mean)
    return np.clip(res, 0, 1)

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
                oldhash = l.pop(0)
                d.pop(oldhash, None)

        return d[hash]
    return newf

@lru_cache
def fib(n):
    if n<=2:
        return 1
    else:
        return fib(n-1)+fib(n-2)


pyramid_kernel = np.array([6, 24, 36, 24, 6], dtype='float32')
pyramid_kernel /= pyramid_kernel.sum()
# @lru_cache
def laplacian_pyramid(img, levels):
    img = zeroone(img)

    gaussian_after_downsize = [img]
    laplacian = []

    for i in range(levels):
        orig = gaussian_after_downsize[-1]
        # blurred = cv2.GaussianBlur(
        #     orig,
        #     ksize=(0,0),
        #     sigmaX=1.0,
        # )

        blurred = cv2.sepFilter2D(
            orig,
            ddepth = cv2.CV_32F,
            kernelX = pyramid_kernel,
            kernelY = pyramid_kernel,
            borderType = cv2.BORDER_REPLICATE,
        )

        laplacian.append(orig - blurred)

        downsized = vis.resize_nearest(
            blurred,
            blurred.shape[0]//2, blurred.shape[1]//2)

        gaussian_after_downsize.append(downsized)

    laplacian.append(blurred)
    return laplacian

if __name__ == '__main__':
    print(fib(50))

    im = load_image('jeff.jpg')
    ai = artistic_enhance(im)

    cv2.imshow('ai', ai)

    im = np.divide(im, 255., dtype='float32')
    lp = laplacian_pyramid(im, levels=4)
    for id,i in enumerate(lp):
        cv2.imshow(str(id), i)
        print(id, i.shape, i.dtype, i.max(), i.min())
    cv2.waitKey(0)
