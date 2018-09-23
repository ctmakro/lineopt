import cv2
import numpy as np
from image import load_image, resize

# see
# https://github.com/ctmakro/face-recognition/blob/master/face_detection_recognition.ipynb

def api(cv2img):
    retval, encoded_jpg = cv2.imencode('.jpg', cv2img, params=(
        cv2.IMWRITE_JPEG_QUALITY, 40
    ))

    encoded_jpg = encoded_jpg.tobytes()

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

def facecrop(img):
    j = api(img)

    if len(j['faces'])>0:
        rect = j['faces'][0]['face_rectangle']

        top, left, width, height =[
            rect[k] for k in ['top','left','width','height']]

        return img[top:top+height, left:left+width]
    else:
        return img

def heatmap(img):
    j = api(img)
    map = np.full(img.shape[0:2]+(1,), 10, dtype='uint8')

    faces = j['faces']

    if __name__ == '__main__':
        # debug purposes
        print(faces[0])
        for k in sorted(faces[0]['landmark']):
            print(k)

    def extract_and_fill(keyword, color, nkeyword=None):
        nkeyword = '$$$' if nkeyword is None else nkeyword
        contour = np.array([
            (v['x'],v['y']) for k,v in landmark.items()
            if (keyword in k) and (nkeyword not in k)
        ])
        contour = cv2.convexHull(contour)
        cv2.fillConvexPoly(
            map,
            contour,
            color = color,
            lineType = cv2.LINE_AA,
        )

    for face in faces:
        landmark = face['landmark']

        for a in [
            ('contour', 20),
            ('left_eye_', 255),
            ('right_eye_', 255),
            ('left_eyebrow', 60),
            ('right_eyebrow', 60),
            ('mouth', 90),
            ('nose_c', 100), # nose contour
            ('nose', 200, 'contour'), # nose keypoints NOT contour
        ]:
            extract_and_fill(*a)

        # draw jawline
        cv2.polylines(
            map,
            [np.array([
                [landmark[name]['x'], landmark[name]['y']]
                for name in [
                    'contour_right6',
                    'contour_right7',
                    'contour_right8',
                    'contour_right9',
                    'contour_chin',
                    'contour_left9',
                    'contour_left8',
                    'contour_left7',
                    'contour_left6',
                ]
            ])],
            isClosed = False,
            color=(60,60,60),
            thickness = 8,
            lineType = cv2.LINE_AA,
            shift = 0,
        )

    def blurit(map):
        blur_r = int(map.shape[0] / 35)
        for i in range(3):
            map = cv2.blur(map, (blur_r, blur_r))
        map.shape+=1,
        return map

    map = blurit(map)

    return np.divide(map, 255, dtype='float32')

if __name__ == '__main__':

    jeff = load_image('jeff.jpg')
    jeff = facecrop(jeff)
    jeff = resize(jeff, 256)

    map = heatmap(jeff)

    cv2.imshow('jeff', jeff)
    cv2.imshow('map', map)

    cv2.imshow('blended', (jeff * map).astype('uint8'))
    cv2.waitKey(0)
