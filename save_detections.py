from datetime import datetime 
st = datetime.now() 
import skimage.io
import os, sys
from os.path import isfile, join
from os import listdir
import numpy as np

ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
import trcnn.model as tracker

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

import pickle

def keepClasses(r, classes, class_names):
	'''Keep only relevant classes in detection step
	Detector can be trained to a superset of relevant classes
	such as car, pedestrian etc'''

	# Extract relevant class_indices
	class_indices = [class_names.index(c) for c in classes]

	# Get all indices of relevant classes
	ri = [ i for i in range(len(r['class_ids'])) if r['class_ids'][i] in class_indices]

	print('metas')
	print(r['detections'].shape)

	# Sample results (r) that match relevant indices
	return {
        "rois": r['rois'][ri,:],
        "class_ids": r['class_ids'][ri],
        "scores": r['scores'][ri],
        "masks": r['masks'][:,:,ri],
        "fp_maps": r['fp_maps'],
        "metas": r['metas'],
        "images": r['images'],
        "detections": r['detections'][:,ri,:]
	}


print("Import: {} (hh:mm:ss.ms)".format(datetime.now()-st))
st = datetime.now()

input_dir = '/home/anthony/maskrcnn/Mask_RCNN/datasets/training/image_02/0014'

# Relevant classes
classes_det = ['Car', 'Pedestrian']

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Output tracking file name
trackf = input_dir[-4:] + '.txt'

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")


class InferenceConfig(coco.CocoConfig):
	# Set batch size to 1 since we'll be running inference on
	# one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

config = InferenceConfig()

# Create model object in inference mode.
model = tracker.TrackRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
# Index of the class in the list is its ID. 
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

# Match KITTI class names
class_names[class_names.index('person')] = 'Pedestrian'
class_names[class_names.index('car')]    = 'Car'

# Read frame names from folder
IMAGE_DIR = input_dir
frames = sorted([f for f in listdir(IMAGE_DIR) if isfile(join(IMAGE_DIR, f))])

print("Model import: {} (hh:mm:ss.ms)".format(datetime.now()-st))
st = datetime.now()

jj = -1
detection_data = []
for frame in frames:
	jj += 1
	print("Frame {}".format(jj))

	# read frame
	st = datetime.now() 
	image = skimage.io.imread(os.path.join(IMAGE_DIR,frame))
	print("Load Image: {} (hh:mm:ss.ms)".format(datetime.now()-st))
	st = datetime.now() 

	extra_boxes =  np.zeros((2,4))
	results = model.detect([image], extra_boxes, verbose=1)
	r = results[0]
	r = keepClasses(r, classes_det, class_names)

	print("Detections: {} (hh:mm:ss.ms)".format(datetime.now()-st))

	st = datetime.now()

	with open('detections_'+ input_dir[-4:] +'_'+str(jj).zfill(6)+'.pickle', 'wb') as f:
	    # Pickle the 'data' dictionary using the highest protocol available.
	    pickle.dump(r, f, pickle.HIGHEST_PROTOCOL)

	print("Pickling detections: {} (hh:mm:ss.ms)".format(datetime.now()-st))




