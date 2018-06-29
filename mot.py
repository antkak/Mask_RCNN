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
from scipy.misc import imresize

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

def assignment(peek_matrix):
	# https://docs.scipy.org/doc/scipy-0.19.1/reference/generated/scipy.optimize.linear_sum_assignment.html
	return 0

def box_dist(bbox1, bbox2):
	# use some metric for box distance
	# center distance, min corner point distance, IoU, IoM etc
	# here I use min corner point distance (due to occlusions)
	c1 = [[bbox1[0], bbox1[1]],
		  [bbox1[0], bbox1[3]],
		  [bbox1[2], bbox1[1]],
		  [bbox1[2], bbox1[3]]]
	c2 = [[bbox2[0], bbox2[1]],
		  [bbox2[0], bbox2[3]],
		  [bbox2[2], bbox2[1]],
		  [bbox2[2], bbox2[3]]]
	distances = [(a[0]-b[0])**2+(a[1]-b[1])**2 for a,b in zip(c1,c2)]
	return min(distances)

def apply_mask(image, mask):
    """Apply the given mask to the image.
    """
    for c in range(256):
        image[:, :, :, c] = np.where(mask == 0,
                                  image[:, :, :, c] * 0 + 0.00001,
                                  image[:, :, :, c])
    return image

def mask_features(FT, metas, mask):
	
	# find image shape using metas
	h = metas[0][1]
	w = metas[0][2]
	
	# pad mask in image square shape
	mask = mask.astype(int)

	if h > w:
		pd = int(h-w)  
		pd1 = pd//2
		pd2 = pd - pd1
		mask = np.pad(mask, ((0,0), (pd1,pd2)), 'constant', constant_values = (0,0))

	else: 
		pd = int(w-h)  
		pd1 = pd//2
		pd2 = pd - pd1
		mask = np.pad(mask, ((pd1,pd2), (0,0)), 'constant', constant_values = (0,0))

	mask = (imresize(mask,(256,256)).astype(bool)).astype(int)
	MT = []
	for T in FT:
		# find T shape
		assert(T.shape[1]==T.shape[2])
		scale = T.shape[1]
		# rescale mask in T shape
		mask_res = (imresize(mask,(scale,scale)).astype(bool)).astype(int)
		# apply mask over all channels
		MT += [apply_mask(T, mask_res)]	
	
	return MT

print("Import: {} (hh:mm:ss.ms)".format(datetime.now()-st))

input_dir =  '/home/anthony/maskrcnn/Mask_RCNN/datasets/training/image_02/0017'
use_extra_boxes = False

# def demo_mot(input_dir, use_extra_boxes=False):
'''
This function solves the MOT problem for the KITTI video sequence
kept in input dir folder
'''

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

# print log info
print("frame {}".format(0))

# read first image
image = skimage.io.imread(os.path.join(IMAGE_DIR,frames[0]))

# run detection
results = model.detect([image], np.zeros((2,4)), verbose=0)

# get results of the first image (batch size is one)
r = results[0]

# keep only relevant classes
r = keepClasses(r, classes_det, class_names)

# Initialize model for Appearance features for boxes
roi_model = tracker.RoiAppearance(config=config)

# Read detections from model output (detections contain rois in normalized coords)
rois = r['detections'][:,:,0:4]

# Run roi encoding for appearance description
appearance = []
for i in range(len(r['class_ids'])):
	# Mask feature tensors (detection are pixel sets, not bounding boxes)
	M0, M1, M2, M3 = mask_features(
		[r['fp_maps'][0],r['fp_maps'][1], r['fp_maps'][2],r['fp_maps'][3]], 
		r['metas'], r['masks'][:,:,i])

	roi_i = np.expand_dims(rois[:,i,:], axis=0)

	appearance += [roi_model.rois_encode(roi_i, r['metas'], M0, M1, M2, M3 )]
				
lost_obj = []
obj_list = []
track_id = 0

# for each object initialize trackedObject
for i in range(len(r['class_ids'])):

	# for simple demo: svd vector representation of feature map appearance
	#app_i = appearance[0,i,:,:,:]
	app_i = np.squeeze(appearance[i], axis=0)	
	app_i = np.squeeze(app_i,axis=0)	
	s = np.linalg.svd(np.transpose(app_i, (2, 0, 1)), full_matrices=False, compute_uv=False)
	app_v = np.transpose(s).flatten('F')

	obj_list += [tob(track_id, r['masks'][:,:,i], r['rois'][i,:],
					r['class_ids'][i], app_v)]
	track_id += 1

# save first frame
save_instances(image, r['rois'], r['masks'], r['class_ids'], 
					class_names, ids = [str(x.id)[:4] for x in obj_list], 
					file_name = str(0)+'.png', colors=[x.color for x in obj_list])

# delete file contents if exist
open(trackf, 'w').close()
# write first frame track results
with open(trackf, 'a') as trf:
	for obj in obj_list:
		if obj.tracking_state == 'New' or obj.tracking_state == 'Tracked':
			trf.write("{} {} {} 0 0 -10.0 {} {} {} {} -1000.0 -1000.0 -1000.0 -10.0 -1 -1 -1 {}\n".format(
						0, obj.id, class_names[obj.class_name], 
						float(obj.bbox[1]), float(obj.bbox[0]), float(obj.bbox[3]), float(obj.bbox[2]), 1))


# for each following frame, run the (classic) MOT algorithm
jj = 0
for frame in frames[1:]:
	jj += 1
	print("Frame {}".format(jj))

	# read frame
	st = datetime.now() 
	image = skimage.io.imread(os.path.join(IMAGE_DIR,frame))
	print("Load Image: {} (hh:mm:ss.ms)".format(datetime.now()-st))
	st = datetime.now() 

	#######################
	### PREDICTION STEP ###
	#######################
	# For each object a (2D image) velocity and (2D image)location is computed
	# Models for motion prediction include Kalman Filtering, Constant velocity
	# assumption, but can also be modeled with RNNs (TODOs). Here we use the CVA
	for obj in obj_list:
		obj.motion_prediction()

	# TODO: Add extra anchors from previous step
	# it seems that extra anchors are not so important from this step due to
	# the covering of anchors of Faster RCNN
	extra_boxes = np.zeros((2,4)) if not use_extra_boxes else \
		np.array([obj.bbox_pred for obj in obj_list])
	
	# TODO: Gating

	#########################################
	#### DETECTION & NEW OBJECT ENCODING ####
	#########################################
	


	results = model.detect([image], extra_boxes, verbose=1)
	r = results[0]
	r = keepClasses(r, classes_det, class_names)

	print("Detections: {} (hh:mm:ss.ms)".format(datetime.now()-st))
	st = datetime.now() 

	# Run roi encoding for appearance description
	rois = r['detections'][:,:,0:4]
	# appearance = roi_model.rois_encode(rois,r['metas'],r['fp_maps'][0],r['fp_maps'][1],
	# 									r['fp_maps'][2],r['fp_maps'][3])
	
	# Run roi encoding for appearance description
	appearance = []
	for i in range(len(r['class_ids'])):
		# Mask feature tensors (detection are pixel sets, not bounding boxes)
		M0, M1, M2, M3 = mask_features(
			[r['fp_maps'][0],r['fp_maps'][1], r['fp_maps'][2],r['fp_maps'][3]], 
			r['metas'], r['masks'][:,:,i])

		roi_i = np.expand_dims(rois[:,i,:], axis=0)

		appearance += [roi_model.rois_encode(roi_i, r['metas'], M0, M1, M2, M3 )]

	print("Appearance encoding: {} (hh:mm:ss.ms)".format(datetime.now()-st))
	st = datetime.now() 	

	# for each newly found object initialize trackedObject
	temp_list = []
	temp_scores = []
	for i in range(len(r['class_ids'])):
	
		# from detector read appearance encoding (from correct pyramid layer)
		# app_i = appearance[0,i,:,:,:]
		# or
		app_i = np.squeeze(appearance[i],axis=0)	
		app_i = np.squeeze(app_i,axis=0)

		s = np.linalg.svd(np.transpose(app_i, (2, 0, 1)), full_matrices=False, compute_uv=False)
		app_v = np.transpose(s).flatten('F')

		# initialize tracked Objects for this current frame
		temp_list += [tob(uuid.uuid4(), r['masks'][:,:,i], r['rois'][i,:],
						r['class_ids'][i], app_v)]
		temp_scores += [r['scores'][i]]

	print("Tracked Object initialization: {} (hh:mm:ss.ms)".format(datetime.now()-st))
	st = datetime.now()

	print('scores {}'.format(temp_scores))

	##########################
	#### DATA ASSOCIATION ####
	##########################
	# Scores for matching objects (rows) to new objects (columns) 

	# APPEARANCE SIMILARITY #
	# Compare old and new objects' appearance 
	peek_matrix = np.zeros((len(obj_list),len(temp_list)))
	for i in range(len(obj_list)):
		for j in range(len(temp_list)):
			peek_matrix[i,j] = simple_dist(obj_list[i],temp_list[j])

	# pad cost matrix if more old objects than new objects
	if peek_matrix.shape[0] > peek_matrix.shape[1]:
		peek_matrix = squarify(peek_matrix, 100)

	# # MOTION CONGRUENCE #
	# # Compare new objects' bounding boxes location to
	# # old objects' predicted bounding boxes location
	# motion_matrix = np.zeros((len(obj_list),len(temp_list)))
	# for i in range(len(obj_list)):
	# 	for j in range(len(temp_list)):
	# 		motion_matrix[i,j] = box_dist(obj_list[i].bbox_pred, temp_list[j].bbox)
	# print(motion_matrix)

	# # pad cost matrix if more old objects than new objects
	# if motion_matrix.shape[0] > motion_matrix.shape[1]:
	# 	motion_matrix = squarify(motion_matrix, 100)

	# run assignment (appearance model)
	row_ind, col_ind = linear_sum_assignment(peek_matrix)
	# # run assignment (motion model)
	# row_ind, col_ind = linear_sum_assignment(motion_matrix)

	print("Assignment problem solving: {} (hh:mm:ss.ms)".format(datetime.now()-st))
	st = datetime.now() 	

	# log assignment
	print(row_ind)
	print(col_ind)

	num_old = len(obj_list)
	num_new = len(temp_list)
	temp_matched = [False]*num_new
	# print(temp_matched)

	########################
	#### UPDATE OBJECTS ####
	########################

	# propagate previous objects in new frame
	for i in range(num_old):
		# if there is a match (old>temp)
		# TODO: treshold matching score
		if col_ind[i] < num_new:
			# refress data
			obj_list[i].refresh_state(True)
			obj_list[i].update_motion(temp_list[col_ind[i]].bbox)
			obj_list[i].mask = temp_list[col_ind[i]].mask
			obj_list[i].encoding = temp_list[col_ind[i]].encoding
			obj_list[i].class_name = temp_list[col_ind[i]].class_name
			temp_matched[col_ind[i]] = True
		# if there is no match, this object is occluded in this frame (or lost if it 
		# is occluded for more than N frames)
		else:
			obj_list[i].refresh_state(False)
			obj_list[i].update_motion(None)

	# initialize new objects
	for i in range(num_new):
		if not temp_matched[i] and temp_scores[i] >= 0.7:
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
		# boxes[i,:] = obj_list_fr[i].bbox_pred
		boxes[i,:] = obj_list_fr[i].bbox
		masks[:,:,i] = obj_list_fr[i].mask

	# save current frame with found objects
	save_instances(image, boxes, masks, [x.class_name for x in obj_list_fr], 
					class_names, ids = [str(x.id)[:4] for x in obj_list_fr], 
					file_name = str(jj)+'.png',colors=[x.color for x in obj_list_fr])
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

#return obj_list



# if __name__ == '__main__':
# 	input_dir =  '/home/anthony/maskrcnn/Mask_RCNN/datasets/training/image_02/0017'
# 	r = demo_mot(input_dir, use_extra_boxes = False)
# 	print(r)

# M0t = np.squeeze(M0,axis=0)



