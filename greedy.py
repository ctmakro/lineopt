# give greedy a try

from image import load_image, resize


class GreedyEnv:
    def set_reference_image(self, img):
        self.ref = img

    def load_reference_image(self, path):
        loaded = load_image(path)
        loaded = resize(loaded, 256)

        self.set_reference_image()

    def __init__(self):
        self.load_reference_image('jeff.jpg')

        self.canvas =

    def new_canvas(self):
