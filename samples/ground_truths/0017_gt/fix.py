
import skimage.io
import os, sys
from os.path import isfile, join
from os import listdir
from skimage.io import imsave

IMAGE_DIR = './'
frames = sorted([f for f in listdir(IMAGE_DIR) if isfile(join(IMAGE_DIR, f))])

for frame in frames:

	# KITTI
	image = skimage.io.imread(os.path.join(IMAGE_DIR,frame))[630:991,210:1446,:]
	# CAVIAR
	# image = skimage.io.imread(os.path.join(IMAGE_DIR,frame))[363:363+884,227:227+1183,:]

	imsave('f'+frame,image)



