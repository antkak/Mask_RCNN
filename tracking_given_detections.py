import skimage.io
import os, sys
from os.path import isfile, join
from os import listdir
import uuid
import numpy as np
import pickle

from scipy.optimize import linear_sum_assignment

ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from trcnn.utils import squarify, box_dist, bbs, sample_boxes, best_buddies_assignment
						
import trcnn.model as tracker
from trcnn.model import trackedObject as tob
from trcnn.model import normalize_boxes

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# Import measurement for tracking 
from measurements import  save_instances

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

	# Log some info
	logf = input_dir[-4:] + 'log' + '.txt'
	# delete file contents if exist
	open(logf, 'w').close()
	lgf = open(logf, 'a')

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

	class InferenceConfig(coco.CocoConfig):
		# Set batch size to 1 since we'll be running inference on
		# one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
		POOL_SIZE = 5
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
		
	# Compute particle bounding boxes and pyramid levels
	pyr_levels, bboxes2_batch, split_list, image = sample_boxes(r, image=image)

	# Keep a copy of absolute coordinates
	bboxes2_abs_batch = bboxes2_batch.copy()

	# Normalize coordinates for encoding
	bboxes2_batch = np.array([normalize_boxes(bboxes2_batch, image.shape[:2])])

	# Encode particle boxes
	app2 = roi_model.rois_encode(bboxes2_batch,r['metas'],r['fp_maps'][0],r['fp_maps'][1],
				r['fp_maps'][2],r['fp_maps'][3])

	# vectorize feature roi pooled maps
	app2_list = [app2[0,i,:,:,:].flatten('F') for i in range(app2.shape[1])]
	app2 = np.array(app2_list)

	# append features to feature list
	# st_i = 1 because bboxes2_abs_batch first row is dummy (zero initialization)
	st_i = 1
	feat_sets = []
	for i in range(len(split_list)):

		feat_sets += [[bboxes2_abs_batch[st_i:split_list[i]+st_i,:], app2[st_i:split_list[i]+st_i,:]/np.linalg.norm(app2[st_i:split_list[i]+st_i,:], axis = 1)[:,None],\
					np.ones(len(app2[st_i:split_list[i]+st_i,:]))]]
		st_i += split_list[i]

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
	patches = []
	for frame in frames[1:]:
		jj += 1
		print("Frame {}".format(jj))
		lgf.write("Frame {}\n".format(jj))

		# read frame
		image = skimage.io.imread(os.path.join(IMAGE_DIR,frame))

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

		# Compute particle bounding boxes and pyramid levels
		pyr_levels, bboxes2_batch, split_list, image = sample_boxes(r, image=image)

		# Keep a copy of absolute coordinates
		bboxes2_abs_batch = bboxes2_batch.copy()

		# Normalize coordinates for encoding
		bboxes2_batch = np.array([normalize_boxes(bboxes2_batch, image.shape[:2])])

		# Encode particle boxes
		app2 = roi_model.rois_encode(bboxes2_batch,r['metas'],r['fp_maps'][0],r['fp_maps'][1],
					r['fp_maps'][2],r['fp_maps'][3])

		# vectorize feature roi pooled maps
		app2_list = [app2[0,i,:,:,:].flatten('F') for i in range(app2.shape[1])]
		app2 = np.array(app2_list)

		# append features to feature list
		# st_i = 1 because bboxes2_abs_batch first row is dummy (zero initialization)
		st_i = 1
		feat_sets = []
		for i in range(len(split_list)):
			feat_sets += [[bboxes2_abs_batch[st_i:split_list[i]+st_i,:], app2[st_i:split_list[i]+st_i,:]/np.linalg.norm(app2[st_i:split_list[i]+st_i,:], axis = 1)[:,None],\
						np.ones(len(app2[st_i:split_list[i]+st_i,:]))]]#[[[bboxes1, app1], [bboxes2, app2]]]
			st_i += split_list[i]


		# for each newly found object initialize trackedObject
		temp_list = []
		temp_scores = []
		for i in range(len(r['class_ids'])):

			# initialize tracked Objects for this current frame
			temp_list += [tob(uuid.uuid4(), r['masks'][:,:,i], r['rois'][i,:],
							r['class_ids'][i], feat_sets[i], pyr_levels[i])]
			temp_scores += [r['scores'][i]]


		##########################
		#### DATA ASSOCIATION ####
		##########################
		# Scores for matching objects (rows) to new objects (columns) 

		# APPEARANCE SIMILARITY #
		# Compare old and new objects' appearance 
		buddy_list = []
		peek_matrix = np.zeros((len(obj_list),len(temp_list)))
		for i in range(len(obj_list)):
			buddy_list_i = []
			for j in range(len(temp_list)):
				# peek_matrix[i,j] = simple_dist(obj_list[i],temp_list[j])
				# if np.abs(obj_list[i].pyramid - temp_list[j].pyramid) > 2:
				# 	peek_matrix[i,j] = 100
				# else:
				bb_sim, bb_b = bbs(obj_list[i],temp_list[j])
				peek_matrix[i,j] = 1-bb_sim
				buddy_list_i += [bb_b]
				# print(buddy_list_i)
				# print('\n')
			buddy_list += [buddy_list_i]
		# print(buddy_list)
		print('\n')

		# pad cost matrix if more old objects than new objects
		if peek_matrix.shape[0] > peek_matrix.shape[1]:
			peek_matrix = squarify(peek_matrix, 100)

		print(peek_matrix)
		lgf.write(str(peek_matrix))
		lgf.write('\n')

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

		# row_ind_, col_ind_ = best_buddies_assignment(peek_matrix)

		lgf.write(str(row_ind))
		lgf.write('\n')
		lgf.write(str(col_ind))
		lgf.write('\n')

		matching_scores = []
		for i in row_ind:
			matching_scores += [peek_matrix[i,col_ind[i]]]
		lgf.write(str(matching_scores))
		lgf.write('\n')


		num_old = len(obj_list)
		num_new = len(temp_list)
		temp_matched = [False]*num_new
		# print(temp_matched)

		########################
		#### UPDATE OBJECTS ####
		########################

		# match_threshold = 0.9
		# propagate previous objects in new frame
		# also save pairs of bounding boxes
		# pairs = []
		for i in range(num_old):
			# if there is a match (old>temp)
			# TODO: treshold matching score
			j = col_ind[i]
			if j < num_new :#and peek_matrix[i,col_ind[i]] < match_threshold:
			# if col_ind[i] < num_new:
				# refress data
				obj_list[i].refresh_state(True)
				obj_list[i].update_motion(temp_list[j].bbox)
				obj_list[i].mask = temp_list[j].mask


				# pairs += [buddy_list[i][j]]

				obj_list[i].refress_encoding(temp_list[j].encoding, buddy_list[i][j])
				obj_list[i].class_name = temp_list[j].class_name
				temp_matched[j] = True
				obj_list[i].score = 1 - peek_matrix[i,j]
				obj_list[i].scores += [[jj, 1 - peek_matrix[i,j]]]


			# if there is no match, this object is occluded in this frame (or lost if it 
			# is occluded for more than N frames)
			else:
				obj_list[i].refresh_state(False)
				obj_list[i].update_motion(None)
		# patches += [pairs]
		# initialize new objects
		det_thresh = 0.7
		for i in range(num_new):
			# for new object initialization the detection score should be >= det_thresh
			if not temp_matched[i] and temp_scores[i] >= det_thresh:
				temp_list[i].id = track_id
				temp_list[i].score = temp_scores[i]
				temp_list[i].scores += [[jj, temp_scores[i]]]
				track_id += 1
				obj_list += [temp_list[i]]

		########################################
		### SAVING RESULTS FOR CURRENT FRAME ###
		########################################

		# keep objects appeared in current frame
		obj_list_fr = [x for x in obj_list if x.tracking_state=='Tracked' or x.tracking_state=='New']
		num_obj = len(obj_list_fr)


		# Prepare object data for saving image
		boxes = np.empty([num_obj,4])
		masks = np.empty([image.shape[0], image.shape[1], num_obj])
		for i in range(num_obj):
			# boxes[i,:] = obj_list_fr[i].bbox_pred
			boxes[i,:] = obj_list_fr[i].bbox
			masks[:,:,i] = obj_list_fr[i].mask

		# save current frame with found objects
		# save_instances(image, boxes, masks, [x.class_name for x in obj_list_fr], 
		# 				class_names, ids = [str(x.id)[:4]+' '+'{:.2f}'.format(x.score) for x in obj_list_fr], 
		# 				file_name = str(jj)+'.png',colors=[x.color for x in obj_list_fr])
		save_instances(image, boxes, masks, [-1 for x in obj_list_fr], 
						class_names, ids = [str(x.id)[:4] for x in obj_list_fr], 
						file_name = str(jj)+'.png',colors=[x.color for x in obj_list_fr])
		# Use this saving code snippet to show pyramid level for debugging
		# save_instances(image, boxes, masks, [x.class_name for x in obj_list_fr], 
		# 				class_names, ids = ['P_'+str(int(np.floor(4+np.log2(1/224*np.sqrt(np.abs((x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))))))) for x in obj_list_fr], 
		# 				file_name = str(jj)+'.png',colors=[x.color for x in obj_list_fr])

		with open(trackf, 'a') as trf:
			for obj in obj_list:
				if obj.tracking_state == 'New' or obj.tracking_state == 'Tracked':
					trf.write("{} {} {} 0 0 -10.0 {} {} {} {} -1000.0 -1000.0 -1000.0 -10.0 -1 -1 -1 {}\n".format(
								jj, obj.id, class_names[obj.class_name], 
								float(obj.bbox[1]), float(obj.bbox[0]), float(obj.bbox[3]), float(obj.bbox[2]), 1))


		# remove lost objects
		lost_obj += [x for x in obj_list if x.tracking_state=='Lost']
		obj_list = [x for x in obj_list if x.tracking_state!='Lost']

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


	return obj_list




if __name__ == '__main__':
	input_dir =  '/home/anthony/maskrcnn/Mask_RCNN/datasets/training/image_02/0014'
	pickle_dir = '/home/anthony/maskrcnn/Mask_RCNN/samples/pickles/0014s'
	# input_dir =  '/home/anthony/mbappe'
	# pickle_dir = '/home/anthony/maskrcnn/Mask_RCNN/samples/pickles/mbappe'
	# input_dir = '/home/anthony/nascar/frames'
	# pickle_dir = '/home/anthony/maskrcnn/Mask_RCNN/samples/pickles/nascar'
	from datetime import datetime 
	st_mot = datetime.now()
	y = [x.encoding for x in demo_mot(input_dir, pickle_dir ,use_extra_boxes = False)]
	print('\n\n\n ++++++++++++++++++\nMOT time: {}'.format(datetime.now()-st_mot))





