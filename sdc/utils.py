import cv2
import matplotlib.image as mpimg
import numpy as np
import os

ORIGINAL_IMAGE_HEIGHT, ORIGINAL_IMAGE_WIDTH = 160, 320
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

class ImageLoader:
    def __init__(self, data_dir, image_paths):
        n_images = image_paths.shape[0]
        self.n_images = n_images

        print('Loading %d images from %s...' % (n_images, data_dir))

        self.center = np.empty([n_images, ORIGINAL_IMAGE_HEIGHT, ORIGINAL_IMAGE_WIDTH, IMAGE_CHANNELS], dtype=np.float32)
        self.left = np.empty([n_images, ORIGINAL_IMAGE_HEIGHT, ORIGINAL_IMAGE_WIDTH, IMAGE_CHANNELS], dtype=np.float32)
        self.right = np.empty([n_images, ORIGINAL_IMAGE_HEIGHT, ORIGINAL_IMAGE_WIDTH, IMAGE_CHANNELS], dtype=np.float32)
        
        for idx, img in enumerate(image_paths):
            center, left, right = img
            self.center[idx] = self._load_image(data_dir, center)
            self.left[idx]   = self._load_image(data_dir, left)
            self.right[idx]  = self._load_image(data_dir, right)

        print('Loaded %d images from %s...' % (n_images, data_dir))

    def _load_image(self, data_dir, image_file):
        return mpimg.imread(os.path.join(data_dir, image_file))

# preprocess an image
def crop(image):
    return image[60:-25, :]

def resize(image):
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)

def rgb2yuv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

def preprocess(image):
    return rgb2yuv(resize(crop(image)))

# augument the image
def choose_image(image_loader, idx, steering_angle):
    choice = np.random.choice(3)
    if choice == 0:
        return image_loader.left[idx], steering_angle + 0.2
    elif choice == 1:
        return image_loader.right[idx], steering_angle - 0.2
    return image_loader.center[idx], steering_angle

def random_flip(img):
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
    return img

def random_translate(img, range_x, range_y):
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    trans_m = np.float32([[1, 0, trans_x],
                          [0, 1, trans_y]])
    h, w = img.shape[:2]
    return cv2.warpAffine(img, trans_m, (w, h))

def random_brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.2 * (np.random.rand() - 0.5)
    hsv[:, :, 2] *= ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def augument(img, range_x=100, range_y=10):
    img = random_flip(img)
    img = random_brightness(img)
    # img = random_translate(img, range_x, range_y)
    return img

def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):
    il = ImageLoader(data_dir, image_paths)

    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty(batch_size)

    while True:
        perm = np.random.permutation(il.n_images)
        i = 0

        # TODO: create one mre generator
        while i < il.n_images:
            bs = np.min([batch_size, il.n_images - i])
            idxs = perm[i:(i + bs)]
            i += bs
            for k, j in enumerate(idxs):
                steering_angle = steering_angles[j]
                if is_training:
                    image, steering_angle = choose_image(il, j, steering_angle)
#                    image = augument(image)
                else:
                    image = il.center[j]
                images[k] = preprocess(image)
                steers[k] = steering_angle
            yield images, steers

