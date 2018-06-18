from datetime import datetime 
st = datetime.now() 
import skimage.io
import os, sys
from os.path import isfile, join
from os import listdir
import uuid
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import correlation
from scipy.optimize import linear_sum_assignment

ROOT_DIR = os.path.abspath("./")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import trcnn.model as tracker
from trcnn.model import trackedObject as tob
from mrcnn import visualize

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# Import measurement for tracking 
from measurements import  save_instances,  save_statistics

print("Import: {} (hh:mm:ss.ms)".format(datetime.now()-st))

def demo_mot(input_dir):

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
	# Index of the class in the list is its ID. For example, to get ID of
	# the teddy bear class, use: class_names.index('teddy bear')
	# class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
	# 			   'bus', 'train', 'truck', 'boat', 'traffic light',
	# 			   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
	# 			   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
	# 			   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
	# 			   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
	# 			   'kite', 'baseball bat', 'baseball glove', 'skateboard',
	# 			   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
	# 			   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
	# 			   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
	# 			   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
	# 			   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
	# 			   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
	# 			   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
	# 			   'teddy bear', 'hair drier', 'toothbrush']
	class_names = ['BG', 'Pedestrian', 'bicycle', 'Car', 'motorcycle', 'airplane',
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

	IMAGE_DIR = input_dir
	frames = sorted([f for f in listdir(IMAGE_DIR) if isfile(join(IMAGE_DIR, f))])

	# read first image
	image = skimage.io.imread(os.path.join(IMAGE_DIR,frames[0]))

	# run detection
	results = model.detect([image], verbose=0)

	# get results of the first image (batch size is one)
	r = results[0]

	# keep only relevant classes
	r = keepClasses(r, classes_det, class_names)

	print(r)

	# Initialize model for Appearance features for boxes
	roi_model = tracker.RoiAppearance(config=config)

	# Read detections from model output (detections contain rois in normalized coords)
	rois = r['detections'][:,:,0:4]
	
	# Run roi encoding for appearance description
	appearance = roi_model.rois_encode(rois,r['metas'],r['fp_maps'][0],r['fp_maps'][1],
		r['fp_maps'][2],r['fp_maps'][3])

	lost_obj = []
	obj_list = []
	track_id = 0
	# for each object initialize trackedObject
	for i in range(len(r['class_ids'])):

		# for simple demo: svd vector representation of feature map appearance
		app_i = appearance[0,i,:,:,:]
		s = np.linalg.svd(np.transpose(app_i, (2, 0, 1)), full_matrices=False, compute_uv=False)
		app_v = np.transpose(s).flatten('F')

		# print(app_v)
		obj_list += [tob(track_id, r['masks'][:,:,i], r['rois'][i,:],
			r['class_ids'][i], app_v)]
		track_id += 1
		print(class_names[r['class_ids'][i]])

	# save first frame
	save_instances(image, r['rois'], r['masks'], r['class_ids'], 
								class_names, scores = [str(x.id)[:4] for x in obj_list], file_name = str(0)+'.png', colors=[x.color for x in obj_list])

	with open(trackf, 'a') as trf:
		for obj in obj_list:
			if obj.tracking_state == 'New' or obj.tracking_state == 'Tracked':
				trf.write("{} {} {} 0 0 -10.0 {} {} {} {} -1000.0 -1000.0 -1000.0 -10.0 -1 -1 -1 {}\n".format(
							0, obj.id, class_names[obj.class_name], 
							float(obj.bbox[1]), float(obj.bbox[0]), float(obj.bbox[3]), float(obj.bbox[2]), 1))

	# print log info
	print("frame {}".format(0))

	# for each following frame
	jj = 0
	for frame in frames[1:]:
		jj += 1
		print("Frame {}".format(jj))
		# read frame
		st = datetime.now() 

		image = skimage.io.imread(os.path.join(IMAGE_DIR,frame))
		print("Load Image: {} (hh:mm:ss.ms)".format(datetime.now()-st))
		st = datetime.now() 


		# TODO: Add extra anchors from previous step
		
		# TODO: Gating

		# Run detection
		results = model.detect([image], verbose=1)

		r = results[0]

		r = keepClasses(r, classes_det, class_names)

		rois = r['detections'][:,:,0:4]

		print("Detections: {} (hh:mm:ss.ms)".format(datetime.now()-st))
		st = datetime.now() 
		# Run roi encoding for appearance description
		appearance = roi_model.rois_encode(rois,r['metas'],r['fp_maps'][0],r['fp_maps'][1],
			r['fp_maps'][2],r['fp_maps'][3])
		print("Appearance encoding: {} (hh:mm:ss.ms)".format(datetime.now()-st))
		st = datetime.now() 	

		# for each newly found object initialize trackedObject
		temp_list = []
		for i in range(len(r['class_ids'])):
		
			# from detector read appearance encoding (from correct pyramid layer)
			app_i = appearance[0,i,:,:,:]
			s = np.linalg.svd(np.transpose(app_i, (2, 0, 1)), full_matrices=False, compute_uv=False)
			app_v = np.transpose(s).flatten('F')

			# initialize tracked Objects for this current frame
			temp_list += [tob(uuid.uuid4(), r['masks'][:,:,i], r['rois'][i,:],
			r['class_ids'][i], app_v)]

			# TODO: motion encoding
		print("Tracked Object initialization: {} (hh:mm:ss.ms)".format(datetime.now()-st))
		st = datetime.now() 	
		"""
		For matching, match old objects (rows) to new objects (columns)
		"""
		# for each old object run simple score and get score matrix
		cost_matrix = np.zeros((len(obj_list),len(temp_list)))
		for i in range(len(obj_list)):
			for j in range(len(temp_list)):
				cost_matrix[i,j] = simple_dist(obj_list[i],temp_list[j])

		# pad cost matrix if more old objects than new objects
		if cost_matrix.shape[0] > cost_matrix.shape[1]:
			cost_matrix = squarify(cost_matrix, 100)

		# run assingment 
		row_ind, col_ind = linear_sum_assignment(cost_matrix)

		print("Assignment problem solving: {} (hh:mm:ss.ms)".format(datetime.now()-st))
		st = datetime.now() 	
		# log assignment
		print(row_ind)
		print(col_ind)

		num_old = len(obj_list)
		num_new = len(temp_list)
		temp_matched = [False]*num_new
		# print(temp_matched)
		# propagate previous objects in new frame
		for i in range(num_old):
			# if there is a match (old>temp)
			# TODO: treshold matching score
			if col_ind[i] < num_new:
				# refress data
				obj_list[i].bbox = temp_list[col_ind[i]].bbox
				obj_list[i].mask = temp_list[col_ind[i]].mask
				obj_list[i].encoding = temp_list[col_ind[i]].encoding
				obj_list[i].class_name = temp_list[col_ind[i]].class_name
				obj_list[i].refress_state(True)
				temp_matched[col_ind[i]] = True
			else:
				obj_list[i].refress_state(False)

		# initialize new objects
		for i in range(num_new):
			if not temp_matched[i]:
				temp_list[i].id = track_id
				track_id += 1
				obj_list += [temp_list[i]]

		# keep objects appeared in current frame
		obj_list_fr = [x for x in obj_list if x.tracking_state=='Tracked' or x.tracking_state=='New']
		num_obj = len(obj_list_fr)

		print("Identity propagation: {} (hh:mm:ss.ms)".format(datetime.now()-st))
		st = datetime.now() 	

		# Prepare object data for saving image
		boxes = np.empty([num_obj,4])
		masks = np.empty([image.shape[0], image.shape[1], num_obj])
		for i in range(num_obj):
			boxes[i,:] = obj_list_fr[i].bbox
			masks[:,:,i] = obj_list_fr[i].mask
		# save current frame with found objects
		save_instances(image, boxes, masks, [x.class_name for x in obj_list_fr], 
									class_names, scores = [str(x.id)[:4] for x in obj_list_fr], file_name = str(jj)+'.png',colors=[x.color for x in obj_list_fr])
		print("Saving Image: {} (hh:mm:ss.ms)".format(datetime.now()-st))
		with open(trackf, 'a') as trf:
			for obj in obj_list:
				if obj.tracking_state == 'New' or obj.tracking_state == 'Tracked':
					trf.write("{} {} {} 0 0 -10.0 {} {} {} {} -1000.0 -1000.0 -1000.0 -10.0 -1 -1 -1 {}\n".format(
								jj, obj.id, class_names[obj.class_name], 
								float(obj.bbox[1]), float(obj.bbox[0]), float(obj.bbox[3]), float(obj.bbox[2]), 1))

		st = datetime.now() 	
		# log object info for current frame
		# for obj in obj_list:
		# 	print("id {}\nstate {}\nbox {}".format(obj.id, obj.tracking_state, obj.bbox))

		# remove lost objects
		lost_obj += [x for x in obj_list if x.tracking_state=='Lost']
		obj_list = [x for x in obj_list if x.tracking_state!='Lost']

	return obj_list

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


def squarify(M,val):
    (a,b)=M.shape
    if a>b:
        padding=((0,0),(0,a-b))
    else:
        padding=((0,b-a),(0,0))
    return np.pad(M,padding,mode='constant',constant_values=val)


def simple_dist(object1, object2):
	# score object1.encoding, object2.encoding
	# multiple encodings, multiple regressing layers

	return correlation(object1.encoding,object2.encoding)

def assignment(cost_matrix):
	# https://docs.scipy.org/doc/scipy-0.19.1/reference/generated/scipy.optimize.linear_sum_assignment.html
	return 0


if __name__ == '__main__':
	input_dir =  '/home/anthony/maskrcnn/Mask_RCNN/datasets/training/image_02/0012'
	print([x.encoding for x in demo_mot(input_dir)])




