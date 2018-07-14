from datetime import datetime 
st = datetime.now() 
import skimage.io
import os, sys
from os.path import isfile, join
from os import listdir
import uuid
import numpy as np
import pickle
import cv2

from scipy.optimize import linear_sum_assignment

ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from trcnn.utils import squarify, keepClasses, simple_dist, box_dist, tensor2vec, \
						pyr_sizes, num_particles, bbs, random_sampling
import trcnn.model as tracker
from trcnn.model import trackedObject as tob
from trcnn.model import normalize_boxes
from mrcnn import visualize

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# Import measurement for tracking 
from measurements import  save_instances,  save_statistics

print("Import: {} (hh:mm:ss.ms)".format(datetime.now()-st))

def demo_mot(input_dir, pickle_dir ,use_extra_boxes=False):
	'''
	This function solves the MOT problem for the KITTI video sequence
	kept in input dir folder
	'''

	pickles = sorted([f for f in listdir(pickle_dir) if isfile(join(pickle_dir, f))])

	# Relevant classes
	classes_det = ['Car', 'Pedestrian']

	# Directory to save logs and trained model
	MODEL_DIR = os.path.join(ROOT_DIR, "logs")

	# Output tracking file name
	trackf = input_dir[-4:] + '.txt'

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

	class InferenceConfig(coco.CocoConfig):
		# Set batch size to 1 since we'll be running inference on
		# one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
		GPU_COUNT = 1
		IMAGES_PER_GPU = 1

	config = InferenceConfig()

	# Read frame names from folder
	IMAGE_DIR = input_dir
	frames = sorted([f for f in listdir(IMAGE_DIR) if isfile(join(IMAGE_DIR, f))])
	
	# Initialize model for Appearance features for boxes
	roi_model = tracker.RoiAppearance(config=config)

	# print log info
	print("frame {}".format(0))
	
	# read first image
	image = skimage.io.imread(os.path.join(IMAGE_DIR,frames[0]))

	## unpickle first dictionary 
	## == run the detector (+4 seconds GPU)
	f = open(join(pickle_dir, pickles[0]),'rb')
	r = pickle.load(f)



	# Read detections from model output (detections contain rois in normalized coords)
	# OR normalize bboxes using tracker.normalize_boxes
	rois = r['detections'][:,:,0:4]
				
	feat_sets = []
	pyr_levels = []

	st = datetime.now()
	for i in range(len(r['class_ids'])):
		
		# feature pyramid that corresponds to object size
		pyramid_level = int(np.floor(4+np.log2(1/224*np.sqrt(np.abs((r['rois'][i,2]-r['rois'][i,0])*(r['rois'][i,3]-r['rois'][i,1]))))))
		pyr_levels += [pyramid_level]

		# run a erosion to get samples inside masks
		kernel = np.array([[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]] ,np.uint8)
		if pyramid_level > 2:
			mask_image = cv2.erode((r['masks'][:,:,i]).astype(np.uint8), kernel, iterations=1).astype(bool)
		else:
			mask_image = r['masks'][:,:,i]
		# mask_image = r['masks'][:,:,i]

		# feature pyramid particle constants that correspond to object size
		M1, M2 = pyr_sizes(pyramid_level) 
		N1, N2 = num_particles(mask_image)

		# visualize particle constants
		# cv2.circle(image, ((r['rois'][i,1]+r['rois'][i,3])//2, (r['rois'][i,0]+r['rois'][i,2])//2), M1//2, (0,0,255), 1)
		# cv2.circle(image, ((r['rois'][i,1]+r['rois'][i,3])//2, (r['rois'][i,0]+r['rois'][i,2])//2), M2//2, (0,0,255), 1)

		# sample points inside mask
		# points1 = random_sampling(mask_image, N1)
		points2 = random_sampling(mask_image, N2)

		M1 = M1//2
		M2 = M2//2
		# points to bounding boxes

		# bboxes1 = np.array([[ point[0]-M1, point[1]-M1, point[0]+M1, point[1]+M1 ] for point in points1]+[[0,0,0,0]])
		bboxes2 = np.array([[ point[0]-M2, point[1]-M2, point[0]+M2, point[1]+M2 ] for point in points2]+[[0,0,0,0]])
		bboxes_abs2 = bboxes2.copy() 

		# until here code is working (tested)


		# normalize bboxes to pass for encoding
		# bboxes1 = np.array([normalize_boxes(bboxes1, image.shape[:2])])
		bboxes2 = np.array([normalize_boxes(bboxes2, image.shape[:2])])

		# encoding
		# endoding can be done once to save time:
		# -> concatenate all bboxes from the various layers, then split the encodings with indexing
		# app1 = roi_model.rois_encode(bboxes1,r['metas'],r['fp_maps'][0],r['fp_maps'][1],
		# 			r['fp_maps'][2],r['fp_maps'][3])
		app2 = roi_model.rois_encode(bboxes2,r['metas'],r['fp_maps'][0],r['fp_maps'][1],
					r['fp_maps'][2],r['fp_maps'][3])

		# vectorize feature roi pooled maps
		# can I do this without lists?
		# app1_list = [app1[0,i,:,:,:].flatten('F') for i in range(app1.shape[1])]
		# app1 = np.array(app1_list)

		app2_list = [app2[0,i,:,:,:].flatten('F') for i in range(app2.shape[1])]
		app2 = np.array(app2_list)

		# remove batch dimension from rois
		# bboxes1 = np.squeeze(bboxes1, axis = 0)
		bboxes2 = np.squeeze(bboxes2, axis = 0)

		# append features to feature list
		feat_sets += [[bboxes_abs2,app2]]#[[[bboxes1, app1], [bboxes2, app2]]]

		# print(bboxes1.shape)
		# print(app1.shape)
		# print(bboxes2.shape)
		# print(app2.shape)

		# visualize bounding boxes
		# for box in bboxes1:
		# 	cv2.rectangle(image,tuple([box[1], box[0]]),tuple([box[3], box[2]]),(0,255,0),1)
		# for box in bboxes2:
		# 	cv2.rectangle(image,tuple([box[1], box[0]]),tuple([box[3], box[2]]),(0,255,0),1)



	# cv2.imshow('im', image)
	# cv2.waitKey(0)
	print("Appearance encoding of particles: {} (hh:mm:ss.ms)".format(datetime.now()-st))

	lost_obj = []
	obj_list = []
	track_id = 0
	# for each object initialize trackedObject
	for i in range(len(r['class_ids'])):

		obj_list += [tob(track_id, r['masks'][:,:,i], r['rois'][i,:],
						r['class_ids'][i], feat_sets[i], pyr_levels[i])]
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
		# for obj in obj_list:
		# 	obj.motion_prediction()
		
		# TODO: Gating

		#########################################
		#### DETECTION & NEW OBJECT ENCODING ####
		#########################################
		
		f = open(join(pickle_dir, pickles[jj]),'rb')
		r = pickle.load(f)

		print("Detections: {} (hh:mm:ss.ms)".format(datetime.now()-st))
		st = datetime.now() 

		# Run roi encoding for appearance description
		rois = r['detections'][:,:,0:4]

	
		feat_sets = []
		pyr_levels = []

		st = datetime.now()
		for i in range(len(r['class_ids'])):
			
			# feature pyramid that corresponds to object size
			pyramid_level = int(np.floor(4+np.log2(1/224*np.sqrt(np.abs((r['rois'][i,2]-r['rois'][i,0])*(r['rois'][i,3]-r['rois'][i,1]))))))
			pyr_levels += [pyramid_level]
				
			kernel = np.array([[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]] ,np.uint8)

			if pyramid_level > 2:
				mask_image = cv2.erode((r['masks'][:,:,i]).astype(np.uint8), kernel, iterations=1).astype(bool)
			else:
				mask_image = r['masks'][:,:,i]
			# feature pyramid particle constants that correspond to object size
			# 1: same pyramid
			# 2: lower pyramid
			M1, M2 = pyr_sizes(pyramid_level) 
			N1, N2 = num_particles(mask_image)

			# visualize particle constants
			# cv2.circle(image, ((r['rois'][i,1]+r['rois'][i,3])//2, (r['rois'][i,0]+r['rois'][i,2])//2), M1//2, (0,0,255), 1)
			# cv2.circle(image, ((r['rois'][i,1]+r['rois'][i,3])//2, (r['rois'][i,0]+r['rois'][i,2])//2), M2//2, (0,0,255), 1)

			# sample points inside mask
			# points1 = random_sampling(mask_image, N1)
			points2 = random_sampling(mask_image, N2)

			M1 = M1//2
			M2 = M2//2
			# points to bounding boxes
			# bboxes1 = np.array([[ point[0]-M1, point[1]-M1, point[0]+M1, point[1]+M1 ] for point in points1]+[[0,0,0,0]])
			# bboxes1_abs1 = bboxes1.copy() 
			bboxes2 = np.array([[ point[0]-M2, point[1]-M2, point[0]+M2, point[1]+M2 ] for point in points2]+[[0,0,0,0]])
			bboxes_abs2 = bboxes2.copy()
			# until here code is working (tested)

			#visualize bounding boxes
			# for i in range(bboxes1.shape[0]):
			# 	cv2.rectangle(image,tuple([bboxes1[i][1], bboxes1[i][0]]),tuple([bboxes1[i][3], bboxes1[i][2]]),(0,255,0),1)
			# for i in range(bboxes2.shape[0]):
			# 	cv2.rectangle(image,tuple([bboxes2[i][1], bboxes2[i][0]]),tuple([bboxes2[i][3], bboxes2[i][2]]),(0,255,0),1)

			# print('Object Characteristics:')
			# print(2*M2)
			# print(pyramid_level)
			# print(np.sum(mask_image))
			# print(N2)
			# normalize bboxes
			# bboxes1 = np.array([normalize_boxes(bboxes1, image.shape[:2])])
			bboxes2 = np.array([normalize_boxes(bboxes2, image.shape[:2])])

			# encoding
			# app1 = roi_model.rois_encode(bboxes1,r['metas'],r['fp_maps'][0],r['fp_maps'][1],
			# 			r['fp_maps'][2],r['fp_maps'][3])
			app2 = roi_model.rois_encode(bboxes2,r['metas'],r['fp_maps'][0],r['fp_maps'][1],
						r['fp_maps'][2],r['fp_maps'][3])

			# vectorize feature roi pooled maps
			# app1_list = [app1[0,i,:,:,:].flatten('F') for i in range(app1.shape[1])]
			# app1 = np.array(app1_list)

			app2_list = [app2[0,i,:,:,:].flatten('F') for i in range(app2.shape[1])]
			app2 = np.array(app2_list)

			# remove batch dimension from rois
			# bboxes1 = np.squeeze(bboxes1, axis = 0)
			bboxes2 = np.squeeze(bboxes2, axis = 0)

			# append features to feature list
			feat_sets += [[bboxes_abs2, app2]]#[[[bboxes1, app1], [bboxes2, app2]]]

			# print(bboxes1.shape)
			# print(app1.shape)
			# print(bboxes2.shape)
			# print(app2.shape)





		# cv2.imshow('im', image)
		# cv2.waitKey(0)
		print("Appearance encoding of particles: {} (hh:mm:ss.ms)".format(datetime.now()-st))

		# print("Appearance encoding: {} (hh:mm:ss.ms)".format(datetime.now()-st))
		st = datetime.now() 	

		# for each newly found object initialize trackedObject
		temp_list = []
		temp_scores = []
		for i in range(len(r['class_ids'])):

			# initialize tracked Objects for this current frame
			temp_list += [tob(uuid.uuid4(), r['masks'][:,:,i], r['rois'][i,:],
							r['class_ids'][i], feat_sets[i], pyr_levels[i])]
			temp_scores += [r['scores'][i]]
		print("Tracked Object initialization: {} (hh:mm:ss.ms)".format(datetime.now()-st))
		st = datetime.now()

		##########################
		#### DATA ASSOCIATION ####
		##########################
		# Scores for matching objects (rows) to new objects (columns) 

		# APPEARANCE SIMILARITY #
		# Compare old and new objects' appearance 
		peek_matrix = np.zeros((len(obj_list),len(temp_list)))
		for i in range(len(obj_list)):
			for j in range(len(temp_list)):
				# peek_matrix[i,j] = simple_dist(obj_list[i],temp_list[j])
				if np.abs(obj_list[i].pyramid - temp_list[j].pyramid) > 2:
					peek_matrix[i,j] = 100
				else:
					peek_matrix[i,j] = 1-bbs(obj_list[i],temp_list[j])
		# pad cost matrix if more old objects than new objects
		if peek_matrix.shape[0] > peek_matrix.shape[1]:
			peek_matrix = squarify(peek_matrix, 100)

		print(peek_matrix)

		#assert(False)
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

		matching_scores = []
		for i in row_ind:
			matching_scores += [peek_matrix[i,col_ind[i]]]
		print(matching_scores)


		num_old = len(obj_list)
		num_new = len(temp_list)
		temp_matched = [False]*num_new
		# print(temp_matched)

		########################
		#### UPDATE OBJECTS ####
		########################

		match_threshold = 100
		# propagate previous objects in new frame
		for i in range(num_old):
			# if there is a match (old>temp)
			# TODO: treshold matching score
			if col_ind[i] < num_new and peek_matrix[i,col_ind[i]] < match_threshold:
			# if col_ind[i] < num_new:
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
		det_thresh = 0.7
		for i in range(num_new):
			# for new object initialization the detection score should be >= det_thresh
			if not temp_matched[i] and temp_scores[i] >= det_thresh:
				temp_list[i].id = track_id
				track_id += 1
				obj_list += [temp_list[i]]

		########################################
		### SAVING RESULTS FOR CURRENT FRAME ###
		########################################

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
		# save_instances(image, boxes, masks, [x.class_name for x in obj_list_fr], 
		# 				class_names, ids = [str(x.id)[:4] for x in obj_list_fr], 
		# 				file_name = str(jj)+'.png',colors=[x.color for x in obj_list_fr])
		# Use this saving code snippet to show pyramid level for debugging
		save_instances(image, boxes, masks, [x.class_name for x in obj_list_fr], 
						class_names, ids = ['P_'+str(int(np.floor(4+np.log2(1/224*np.sqrt(np.abs((x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))))))) for x in obj_list_fr], 
						file_name = str(jj)+'.png',colors=[x.color for x in obj_list_fr])
		print("Saving Image: {} (hh:mm:ss.ms)".format(datetime.now()-st))
		with open(trackf, 'a') as trf:
			for obj in obj_list:
				if obj.tracking_state == 'New' or obj.tracking_state == 'Tracked':
					trf.write("{} {} {} 0 0 -10.0 {} {} {} {} -1000.0 -1000.0 -1000.0 -10.0 -1 -1 -1 {}\n".format(
								jj, obj.id, class_names[obj.class_name], 
								float(obj.bbox[1]), float(obj.bbox[0]), float(obj.bbox[3]), float(obj.bbox[2]), 1))

		st = datetime.now() 	

		# remove lost objects
		lost_obj += [x for x in obj_list if x.tracking_state=='Lost']
		obj_list = [x for x in obj_list if x.tracking_state!='Lost']

	return obj_list

if __name__ == '__main__':
	input_dir =  '/home/anthony/maskrcnn/Mask_RCNN/datasets/training/image_02/0017'
	pickle_dir = '/home/anthony/maskrcnn/Mask_RCNN/samples/pickles/0017'
	print([x.encoding for x in demo_mot(input_dir, pickle_dir ,use_extra_boxes = False)])
	# mask = np.array([[True, True, True],
	# 				 [False, False, True]])
	# print(random_sampling(mask, None))




