import sys
sys.path.append('../')

from image import *
a = lambda *k:np.array(k)

im = load_image('../jeff.jpg')

from facial import facecrop
# im = facecrop(im)
#
# im = vis.resize_perfect(im, 512, 512)


# black, yellow, white
colors = a([0,0,0], [255,220,113], [255,255,255])
colors = np.flip(colors, axis=1)
colors = np.divide(colors, 255., dtype=np.float32)

# print(colors)

def artistic_enhance(img):
    img = zeroone(img)

    blurred = cv2.GaussianBlur(img,ksize=(0,0),sigmaX=20)

    theta = f32(.3)
    alpha = f32(1.2)

    mean = img.mean()

    hf = img - blurred
    res = (img + hf * theta) * alpha + (0.5 - mean)
    return np.clip(res, 0, 1)

ai = artistic_enhance(im)
# ai = np.divide(im, 255., dtype=np.float32)

def separate(img, colors):
    distances = []
    for c in colors:
        distances.append(
            ((img - c) ** 2).sum(axis=-1)
        )

    stacked = np.stack(distances, axis=-1)
    print(stacked.shape)

    argmin = np.argmin(stacked, axis=-1)
    return argmin

map = separate(ai, colors)
map.shape+=(1,)

cmap = np.zeros_like(ai)
for i in range(len(colors)):
    ci = colors[i]
    ci.shape = (1,1) + ci.shape
    cmap += ci * (map==i)

cv2.imshow('cmap', cmap)

cv2.imshow('ai', ai)
cv2.waitKey(0)
