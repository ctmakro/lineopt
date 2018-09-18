import cv2
import numpy as np
from image import load_image, resize

# see
# https://github.com/ctmakro/face-recognition/blob/master/face_detection_recognition.ipynb

def api(encoded_jpg):
    import requests
    # for more detailed information on the API interface,
    # please (after log into their website) refer to
    # https://console.faceplusplus.com/documents/5679127

    # here an HTTP POST request is constructed and sent to faceplusplus.com's API endpoint.
    response = requests.post(
        'https://api-us.faceplusplus.com/facepp/v3/detect',
        # 'https://api-us.faceplusplus.com/facepp/v3/detect',
        data = {
            # here I use my own API key and secret;
            # you can go to faceplusplus.com and register an account
            # to get your own api_key and api_secret for FREE after email verification.
            'api_key':'GKDiPpGXxPZt0QVYCUUhUHyW1XdviF6z',
            'api_secret':'wsxmlqMWsRYTNTeHMYIXd7i7n9P9cSRS',

            # we want the landmark information (position of facial features)
            'return_landmark':1,
        },

        # by specifying the `files` argument below, the `requests`
        # library will construct a `multipart` request, with
        # binary file(s) as part of the request.
        files = {
            # pass in the file object here
            'image_file': encoded_jpg
        }
    )

    import json
    response_json = json.loads(response.text)

    return response_json

def heatmap(img):
    retval, encoded_jpg = cv2.imencode('.jpg', img)
    j = api(encoded_jpg.tobytes())

    assert len(j['faces']) > 0

    face = j['faces'][0]

    map = np.zeros(img.shape[0:2]+(1,), dtype='uint8')

    def extract_and_fill(keyword, color, nkeyword=None):
        nkeyword = '$$$' if nkeyword is None else nkeyword
        contour = np.array([
            (v['x'],v['y']) for k,v in face['landmark'].items()
            if (keyword in k) and (nkeyword not in k)
        ])
        contour = cv2.convexHull(contour)
        cv2.fillConvexPoly(
            map,
            contour,
            color = color,
            lineType = cv2.LINE_AA,
        )

    for a in [
        ('contour', 25),
        ('left_eye_', 255),
        ('right_eye_', 255),
        ('left_eyebrow', 150),
        ('right_eyebrow', 150),
        ('mouth', 150),
        ('nose_c', 100), # nose contour
        ('nose', 200, 'contour'), # nose keypoints NOT contour
    ]:
        extract_and_fill(*a)

    return map

if __name__ == '__main__':

    jeff = load_image('jeff.jpg')
    jeff = resize(jeff, 256)

    map = heatmap(jeff)

    cv2.imshow('jeff', jeff)
    cv2.imshow('map', map)

    cv2.imshow('blended', (jeff*0.5 + map*0.5).astype('uint8'))
    cv2.waitKey(0)
