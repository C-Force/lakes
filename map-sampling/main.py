#!/Users/chess/.virtualenvs/map-samling/bin/python

import glob
import os
from samples import generate_positive_samples, generate_negative_samples, squarelize
import cv2
import argparse

MAX_SIZE = 224
STRIDE = 56

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Path to the maps")
args = vars(ap.parse_args())
root = args['path']

if not os.path.exists(root):
  print('Error: invalid path\n')
  exit()

os.chdir(root)

if not os.path.exists('positive'):
  os.makedirs('positive')
if not os.path.exists('negative'):
  os.makedirs('negative')

files = glob.glob('NIDID*.png')
for idx, filename in enumerate(files):
  total = len(files)
  img = cv2.imread(filename)
  img = squarelize(img)
  positive_samples = generate_positive_samples(img, MAX_SIZE, MAX_SIZE, STRIDE)
  negative_samples = generate_negative_samples(img, 126, (MAX_SIZE, MAX_SIZE), center_bound=STRIDE)
  for i, p in enumerate(positive_samples):
    path = 'positive/' + filename.split('.')[0] + '_{0:0=3d}'.format(i) + '.png'
    print('saving ' + path + '\t\t\t{:.0%}'.format(idx/total))
    cv2.imwrite(path, p)
  for i, n in enumerate(negative_samples):
    path = 'negative/' + filename.split('.')[0] + '_{0:0=3d}'.format(i) + '.png'
    print('saving ' + path + '\t\t\t{:.0%}'.format(idx/total))
    cv2.imwrite(path, n)