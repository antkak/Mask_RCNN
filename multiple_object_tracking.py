import skimage.io
import os, sys
from os.path import isfile, join
from os import listdir
import numpy as np

ROOT_DIR = os.path.abspath("./")

# Import local MOT modules
sys.path.append(ROOT_DIR)  
import trcnn.model as tracker

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

def demo_mot(input_dir):
	'''
	This function solves the MOT problem for the KITTI video sequence
	kept in input dir folder
	'''

	# Local path to trained weights file
	COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
	# Directory to save logs and trained model
	MODEL_DIR = os.path.join(ROOT_DIR, "logs")

	# Output tracking file name
	trackf = input_dir[-4:] + '.txt'
	# Log some info
	logf = input_dir[-4:] + 'log' + '.txt'

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
				   'teddy bear', 'hair drier', 'toothbrush', ' ']
	# Match KITTI class names
	class_names[class_names.index('person')] = 'Pedestrian'
	class_names[class_names.index('car')]    = 'Car'
	# Relevant classes
	classes_det = ['Car', 'Pedestrian']


	class EncoderConfig(coco.CocoConfig):
		# Configuration for bounding box encoding
		POOL_SIZE = 5
		GPU_COUNT = 1
		IMAGES_PER_GPU = 1

	enc_config = EncoderConfig()


	# Configuration class
	class InferenceConfig(coco.CocoConfig):
		# Set batch size to 1 since we'll be running inference on
		# one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
		GPU_COUNT = 1
		IMAGES_PER_GPU = 1

	config = InferenceConfig()

	# Configuration class
	class TrackingConfig():

		def all(N):
			return N
		# parameters for tracking
		# save_detection images
		SAVE_DETECTIONS = False
		# use_boxes
		USE_BOXES = False
		# use_spatial_constraints
		USE_SPATIAL_CONSTRAINTS = True
		# save_spatial_constraints
		SAVE_SPATIAL_CONSTRAINTS = False
		# kalman filter parameters
		KF_Q = np.diag([1,1,10,10])
		KF_P = np.diag([10,10,1000,1000])
		KF_R = np.diag([100,100])
		# Appearance Drift Multiplier
		APP_DRIFT_MULTIPLIER = 0.8
		FRAME_THRESHOLD = 5
		# Number of sampling points function
		# SAMPLING_NUM_FUN = all()
		MATCH_THRESHOLD = 0.9

	track_config = TrackingConfig()

	# Create model object in inference mode and load weights
	detect_model = tracker.TrackRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
	detect_model.load_weights(COCO_MODEL_PATH, by_name=True)

	# Initialize particle set encoder
	encoder = tracker.ParticleDescription(config=enc_config)

	# Summon Deep Appearance Robust Tracker
	dart = tracker.DART(config=track_config, output_file=trackf, class_names=class_names, log_file=logf)


	# Read frame names from folder
	IMAGE_DIR = input_dir
	frames = sorted([f for f in listdir(IMAGE_DIR) if isfile(join(IMAGE_DIR, f))])
	frame_num = 0

	# wait until objects are detected
	while frame_num < len(frames):
		
		# print log info
		print("frame {}".format(frame_num))
		
		# read first image
		image = skimage.io.imread(os.path.join(IMAGE_DIR,frames[frame_num]))

		# run detection
		r = detect_model.detect([image], np.zeros((2,4)), classes_det, class_names, verbose=0)
		frame_num += 1

		if len(r['class_ids'])>0:
			break

	# appearance encoding for each detection
	feat_sets, pyr_levels = encoder.encode(r, image)

	# initialize tracker with first detections
	dart.initialize(r, feat_sets, pyr_levels, image, frame=0)
	
	# for each following frame, run the (classic) MOT algorithm
	for frame in frames[frame_num:]:

		# read frame
		image = skimage.io.imread(os.path.join(IMAGE_DIR,frame))

		# run detection
		r = detect_model.detect([image], np.zeros((2,4)), classes_det, class_names, verbose=0)

		# if no detections, next frame
		if len(r['class_ids']) == 0:
			frame_num += 1
			continue

		# appearance encoding
		feat_sets, pyr_levels = encoder.encode(r, image)

		# data association and object updates
		dart.associate(r, feat_sets, pyr_levels, image, frame_num)

		frame_num += 1

	# # code for best buddies visualization
	# import cv2

	# for i, patch in zip(range(len(frames)), patches):
	# 	img1 = skimage.io.imread(os.path.join(IMAGE_DIR,frames[i]))
	# 	img2 = skimage.io.imread(os.path.join(IMAGE_DIR,frames[i+1]))
	# 	h,w,_ = img1.shape
	# 	for p in patch:
	# 		image = np.concatenate((img1, img2), axis=1)
	# 		for pair in p:
	# 			cv2.rectangle(image,tuple([pair[0][1], pair[0][0]]),tuple([pair[0][3],   pair[0][2]]),(0,255,0),1)
	# 			cv2.rectangle(image,tuple([pair[1][1]+w, pair[1][0]]),tuple([pair[1][3]+w, pair[1][2]]),(0,255,0),1)
	# 			cv2.line(image, tuple([pair[0][1], pair[0][0] ]), tuple([pair[1][1]+w, pair[1][0]]), (0,255,0), 1)
	# 		cv2.imshow('im', image)
	# 		cv2.waitKey(0)
	# 	if i > 2:
	# 		break

	# code for saving all tracked objects 
	# with open('objects.pickle', 'wb') as f:
	# 	# Pickle the 'data' dictionary using the highest protocol available.
	# 	pickle.dump(obj_list+lost_obj, f, pickle.HIGHEST_PROTOCOL)


	return dart.obj_list




if __name__ == '__main__':
	input_dir =  '/home/anthony/maskrcnn/Mask_RCNN/datasets/training/image_02/0014'
	# input_dir =  '/home/anthony/mbappe'
	# input_dir = '/home/anthony/nascar/frames'
	from datetime import datetime 
	st_mot = datetime.now()
	y = [x.encoding for x in demo_mot(input_dir)]
	print('\n\n\n ++++++++++++++++++\nMOT time: {}'.format(datetime.now()-st_mot))

