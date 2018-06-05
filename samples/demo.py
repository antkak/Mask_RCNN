import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
	utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
#IMAGE_DIR = os.path.join(ROOT_DIR, "images")
IMAGE_DIR = os.path.join(ROOT_DIR,'datasets/testing/image_02/0001/')



class InferenceConfig(coco.CocoConfig):
	# Set batch size to 1 since we'll be running inference on
	# one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
			   'bus', 'train', 'truck', 'boat', 'traffic light',
			   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
			   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
			   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
			   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
			   'kite', 'baseball bat', 'baseball glove', 'skateboard',
			   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
			   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
			   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
			   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
			   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
			   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
			   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
			   'teddy bear', 'hair drier', 'toothbrush']

from measurements import save_instances

# Load a random image from the images folder
file_names = [f for f in listdir(IMAGE_DIR) if isfile(join(IMAGE_DIR, f))]
for name in file_names:
	image = skimage.io.imread(os.path.join(IMAGE_DIR,name))

	# Run detection
	results = model.detect([image], verbose=1)

	# Visualize results
	r = results[0]
	save_instances(image, r['rois'], r['masks'], r['class_ids'], 
								class_names, r['scores'], file_name = name)