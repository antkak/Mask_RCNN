'''
Code for KITTI benchmark evaluation
Detections are given as a file

### ENCODING ###
Frame forward pass to extract feature maps
Bounding boxes ROI pooling in feature map
ROI pooled regions to mask branch to extract pixel mask
Use pixel mask to sample patches
Patches pooling and tensor2vec

'''
import skimage.io
import numpy as np

import os, sys
from os.path import isfile, join
from os import listdir

ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)  # To find local version of the library

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

import trcnn.model as tracker

from measurements import  save_instances,  save_statistics


def tracking_evaluation(FRAME_DIR, DETECTION_DIR):

	# Read frame names
	IMAGE_DIR = FRAME_DIR
	frames = sorted([f for f in listdir(IMAGE_DIR) if isfile(join(IMAGE_DIR, f))])
	
	# read first image
	image = skimage.io.imread(os.path.join(IMAGE_DIR,frames[0]))

	imshape = image.shape[:2]

	frame_cnt = 1

	# Read Detection File
	with open(DETECTION_DIR, 'r') as detf:

		lines = [line.split(',') for line in detf.readlines()]

	current_lines = [x for x in lines if x[0]==str(frame_cnt)]

	cr = 0.05

	rois = np.array([[float(x[3])-cr*float(x[5]),float(x[2])-cr*float(x[4]),float(x[3])+(1+cr)*float(x[5]),float(x[2])+(1+cr)*float(x[4])] for x in current_lines])

	rois = np.append(rois,np.zeros((100-rois.shape[0],4)), axis=0)

	print(rois.shape)

	# Output Tracking File name
	trackf = FRAME_DIR[-4:] + '.txt'

	# Directory to save logs and trained model
	MODEL_DIR = os.path.join(ROOT_DIR, "logs")

	# Local path to trained weights file
	COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
	
	# Configuration class
	class InferenceConfig(coco.CocoConfig):
		# Set batch size to 1 since we'll be running inference on
		# one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
		DETECTION_MIN_CONFIDENCE = 0
		GPU_COUNT = 1
		IMAGES_PER_GPU = 1

	config = InferenceConfig()

	# Create model object in inference mode.
	model = tracker.TrackRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

	# Load weights trained on MS-COCO
	model.load_weights(COCO_MODEL_PATH, by_name=True)

	# print log info
	print("frame {}".format(frame_cnt))

	# run detection
	results = model.detect([image], np.zeros((2,4)), verbose=0)

	# get feature maps of the first image (batch size is one)
	fp_maps = results[0]['fp_maps']

	# Extract masks from fp_maps and detection boxes
	
	# Initialize heads
	heads = tracker.MaskrcnnHeads(config=config)

	heads.load_weights(COCO_MODEL_PATH, by_name=True)

	results = heads.masked([image], config, rois, fp_maps, imshape)

	r = results[0]

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
				   'teddy bear', 'hair drier', 'toothbrush', ' ']

	# save first frame
	save_instances(image, r['rois'], r['masks'], r['class_ids'], 
						class_names, ids = [1 for i in range(len(r['rois']))], 
						file_name = str(frame_cnt)+'.png', colors=[[1,1,1] for i in range(len(r['rois']))])

	for frame in frames[2:21]:

		image = skimage.io.imread(os.path.join(IMAGE_DIR,frame))
		frame_cnt += 1
		# print log info
		print("frame {}".format(frame_cnt))
		current_lines = [x for x in lines if x[0]==str(frame_cnt)]

		rois = np.array([[float(x[3]),float(x[2]),float(x[3])+float(x[5]),float(x[2])+float(x[4])] for x in current_lines])

		rois = np.append(rois,np.zeros((100-rois.shape[0],4)), axis=0)


		# run detection
		results = model.detect([image], np.zeros((2,4)), verbose=0)

		# get feature maps of the first image (batch size is one)
		fp_maps = results[0]['fp_maps']

		# Extract masks from fp_maps and detection boxes
		

		results = heads.masked([image], config, rois, fp_maps, imshape)

		r = results[0]

		# save first frame
		save_instances(image, rois[:len(r['class_ids'])], r['masks'], r['class_ids'], 
						class_names, ids = [1 for i in range(len(r['rois']))], 
						file_name = str(frame_cnt)+'.png', colors=[[1,1,1] for i in range(len(r['rois']))])
	


if __name__ == '__main__':
	FRAME_DIR = '/home/anthony/maskrcnn/Mask_RCNN/datasets/training/image_02/0017'
	DETECTION_DIR = '/home/anthony/maskrcnn/Mask_RCNN/0017_det.txt'
	tracking_evaluation(FRAME_DIR, DETECTION_DIR)
