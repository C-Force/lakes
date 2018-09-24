"""This script generate square samples from a map image of a lake with the dam at its center"""

from itertools import product
import numpy as np
import cv2

MAX_SIZE = 224
STRIDE = 56
POS_DEGREES = [45, 90, 135]
DEGREES = POS_DEGREES + [-d for d in POS_DEGREES] + [180]

def chunks(l, n):
  """Yield successive n-sized chunks from l."""
  for i in range(0, len(l), n):
    yield l[i:i + n]

def draw_box(img, center, width=100, height=100):
  """generate box around a center"""
  x_1 = center[0] - width//2
  x_2 = center[0] + width//2
  y_1 = center[1] - height//2
  y_2 = center[1] + height//2
  return img[x_1:x_2, y_1:y_2]

def translate_boxes(img, width=100, height=100, stride=10):
  center = (img.shape[0] // 2, img.shape[1] // 2)
  move_to = list(product([stride, 0, -stride], repeat=2))
  moved_boxes = []
  for c in [(center[0]+offset[0], center[1]+offset[1]) for offset in move_to]:
    moved_boxes.append(draw_box(img, c, width, height))
  return moved_boxes

def rotate_image(img):
  rotated_images = []
  center = (img.shape[0] // 2, img.shape[1] // 2)
  for d in DEGREES:
    m = cv2.getRotationMatrix2D(center, d, 1)
    rotated_images.append(cv2.warpAffine(img, m, img.shape[:2]))
  return rotated_images

def flip_boxes(img):
  return [cv2.flip(img, i) for i in [0, 1]]

def generate_positive_samples(img, width, height, stride):
  boxes = []
  for f in flip_boxes(img):
    for r in rotate_image(f):
      for t in translate_boxes(r, width, height, stride):
        boxes.append(t)
  return boxes

def show_images(imgs):
  boxes = list(chunks(imgs, 9))
  hmerge = list(map(np.hstack, boxes))
  cv2.imshow('results', np.vstack(hmerge))
  cv2.waitKey(0)

def check_bounds(img, p, center_bound=30):
  center = (img.shape[0] // 2, img.shape[1] // 2)
  return not (center[0] - center_bound < p[0] < center[0] - center_bound \
    and center[1] - center_bound < p[1] < center[1] - center_bound)

def random_coords(img, random_crop_size, sync_seed=None, center_bound=30):
  np.random.seed(sync_seed)
  w, h = img.shape[0], img.shape[1]
  rangew = (w - random_crop_size[0]) // 2
  rangeh = (h - random_crop_size[1]) // 2
  offsetw = 0 if rangew == 0 else np.random.randint(rangew)
  offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
  return (offsetw, offsetw+random_crop_size[0]), (offseth, offseth+random_crop_size[1])

def random_crop(img, random_crop_size, sync_seed=None, center_bound=30):
  xs, ys = random_coords(img, random_crop_size, sync_seed, center_bound)
  while True:
    if all([check_bounds(img, p, center_bound) for p in list(product(xs, ys))]):
      return img[xs[0]:xs[1], ys[0]:ys[1]]
    xs, ys = random_coords(img, random_crop_size, sync_seed, center_bound)

def squarelize(img):
  if img.shape[0] != img.shape[1]:
    center = (img.shape[0]//2, img.shape[1]//2)
    size = min(img.shape[:2])
    return draw_box(img, center, size, size)
  return img

def generate_negative_samples(img, size, random_crop_size, sync_seed=None, center_bound=30):
  return [random_crop(img, random_crop_size, sync_seed, center_bound) for i in range(size)]

# img = cv2.imread('NIDID_MI00016.png')
# img = squarelize(img)
# center = (img.shape[0]//2, img.shape[1]//2)
# boxes = generate_positive_samples(img, MAX_SIZE, MAX_SIZE, STRIDE)
# show_images(boxes)
# samples = generate_negative_samples(img, 126, (MAX_SIZE, MAX_SIZE), center_bound=STRIDE)
# show_images(samples)