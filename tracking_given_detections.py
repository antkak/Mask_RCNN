import skimage.io
import os, sys
from os.path import isfile, join
from os import listdir
import uuid
import numpy as np
import pickle
import matplotlib.patches as patches
from matplotlib.patches import Ellipse

from scipy.optimize import linear_sum_assignment
from matplotlib import pyplot as plt
import cv2

ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from trcnn.utils import squarify, box_dist, bbs, sample_boxes, best_buddies_assignment, gating, gkern, gating_mask
						
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

	class EncoderConfig(coco.CocoConfig):
		# Configuration for bounding box encoding
		POOL_SIZE = 5
		GPU_COUNT = 1
		IMAGES_PER_GPU = 1

	config = EncoderConfig()

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
	pyr_levels, bboxes_batch, split_list, image = sample_boxes(r, image=image)

	# Keep a copy of absolute coordinates
	bboxes_abs_batch = bboxes_batch.copy()

	# Normalize coordinates for encoding
	bboxes_batch = np.array([normalize_boxes(bboxes_batch, image.shape[:2])])

	# Create normalization
	sigma = config.POOL_SIZE/2.0/3.0
	kerlen = config.POOL_SIZE
	W = gkern(kernlen=kerlen, nsig=sigma)
	w = np.stack([W for i in range(256)]).T.flatten('F')
	print(W.shape)
	print(w.shape)
	kernel = False

	# Encode particle boxes
	app = roi_model.rois_encode(bboxes_batch,r['metas'],r['fp_maps'][0],r['fp_maps'][1],
				r['fp_maps'][2],r['fp_maps'][3])

	if kernel:
		app_list = [np.multiply(app[0,i,:,:,:].flatten('F'),w) for i in range(app.shape[1])]
	else: 
		app_list = [app[0,i,:,:,:].flatten('F') for i in range(app.shape[1])]

	app = np.array(app_list)
	# vectorize feature roi pooled map	app = np.array(app_list)

	# append features to feature list
	# st_i = 1 because bboxes_abs_batch first row is dummy (zero initialization)
	st_i = 1
	feat_sets = []
	for split in split_list:

		feat_sets += [[bboxes_abs_batch[st_i:split+st_i,:], \
				app[st_i:split+st_i,:]/np.linalg.norm(app[st_i:split+st_i,:], axis = 1)[:,None],\
				np.ones(len(app[st_i:split+st_i,:]))]]
		st_i += split

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

		# Prediction Step	
		for obj in obj_list:
			obj.location_prediction()

		# plot uncertainties
		fig = plt.figure() 
		ax1 = fig.add_subplot(111, aspect='equal')
		ax1.imshow(image)
		for obj in obj_list:
			b_gain = np.sqrt((obj.bbox[2]-obj.bbox[0])**2 + (obj.bbox[3]-obj.bbox[1])**2)/np.sqrt(2)/2
			x = obj.x_minus
			P = obj.P_minus
			lambda_, v = np.linalg.eig(P[:2,:2])
			lambda_ = np.sqrt(lambda_)
			j = 3
			ell = Ellipse(xy=(x[0], x[1]),
				width=(lambda_[0]*j+b_gain)*2, height=(lambda_[1]*j+b_gain)*2,
				angle=np.rad2deg(np.arccos(v[0, 0])), fill=False,lw=3)
			ell.set_facecolor('none')
			ax1.add_patch(ell)
		
		# Detections and encoding
		
		f = open(join(pickle_dir, pickles[jj]),'rb')
		r = pickle.load(f)

		# Compute particle bounding boxes and pyramid levels
		pyr_levels, bboxes_batch, split_list, image = sample_boxes(r, image=image)

		# Keep a copy of absolute coordinates
		bboxes_abs_batch = bboxes_batch.copy()

		# Normalize coordinates for encoding
		bboxes_batch = np.array([normalize_boxes(bboxes_batch, image.shape[:2])])

		# Encode particle boxes
		app = roi_model.rois_encode(bboxes_batch,r['metas'],r['fp_maps'][0],r['fp_maps'][1],
					r['fp_maps'][2],r['fp_maps'][3])

		# vectorize feature roi pooled maps
		if kernel:
			app_list = [np.multiply(app[0,i,:,:,:].flatten('F'),w) for i in range(app.shape[1])]
		else: 
			app_list = [app[0,i,:,:,:].flatten('F') for i in range(app.shape[1])]
		app = np.array(app_list)

		# append features to feature list
		# st_i = 1 because bboxes_abs_batch first row is dummy (zero initialization)
		st_i = 1
		feat_sets = []
		for split in split_list:
			feat_sets += [[bboxes_abs_batch[st_i:split+st_i,:], \
					app[st_i:split+st_i,:]/np.linalg.norm(app[st_i:split+st_i,:], axis = 1)[:,None],\
					np.ones(len(app[st_i:split+st_i,:]))]]#[[[bboxes1, app1], [bboxes2, app]]]
			st_i += split


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
		for i,obj in enumerate(obj_list):
			buddy_list_i = []
			for j,temp in enumerate(temp_list):

				bb_sim, bb_b = bbs(obj,temp)
				peek_matrix[i,j] = 1-bb_sim
				buddy_list_i += [bb_b]
				b_gain = np.sqrt((obj.bbox[2]-obj.bbox[0])**2 + (obj.bbox[3]-obj.bbox[1])**2)/np.sqrt(2)/2
				if gating(obj.x_minus, obj.P_minus, temp.x, b_gain):
					peek_matrix[i,j] = 100


			buddy_list += [buddy_list_i]

		fig.canvas.flush_events()
		plt.draw()
		plt.savefig('_'+str(jj)+'.png')
		# plt.show()
		ax1.cla()

		# pad cost matrix if more old objects than new objects
		if peek_matrix.shape[0] > peek_matrix.shape[1]:
			peek_matrix = squarify(peek_matrix, 100)

		print('\n'+str(peek_matrix))

		# run assignment (appearance model)
		row_ind, col_ind = linear_sum_assignment(peek_matrix)

		matching_scores = []
		for i in row_ind:
			matching_scores += [peek_matrix[i,col_ind[i]]]
		lgf.write(str(peek_matrix)+'\n'+str(row_ind)+'\n'+str(col_ind)+'\n'+str(matching_scores)+'\n')

		num_old = len(obj_list)
		num_new = len(temp_list)
		temp_matched = [False]*num_new

		########################
		#### UPDATE OBJECTS ####
		########################

		match_threshold = 0.9
		# propagate previous objects in new frame
		# also save pairs of bounding boxes
		# pairs = []
		for i,obj in enumerate(obj_list):
			j = col_ind[i]
			# if there is a match (old>temp)
			if j < num_new and peek_matrix[i,col_ind[i]] <= match_threshold:
				# refress data
				obj.refresh_state(True)
				obj.mask = temp_list[j].mask

				# pairs += [buddy_list[i][j]]

				obj.refress_encoding(temp_list[j].encoding, buddy_list[i][j])
				obj.class_name = temp_list[j].class_name
				temp_matched[j] = True
				obj.score = 1 - peek_matrix[i,j]
				obj.scores += [[jj, 1 - peek_matrix[i,j]]]
				obj.location_update(obj.x_minus, obj.P_minus, temp_list[j].bbox )


			# if there is no match, this object is occluded in this frame (or lost if it 
			# is occluded for more than N frames)
			else:
				obj.refresh_state(False)
				obj.location_update(obj.x_minus, obj.P_minus, None )


		# patches += [pairs]
		# initialize new objects
		det_thresh = 0.7
		for i, temp in enumerate(temp_list):
			# for new object initialization the detection score should be >= det_thresh
			if not temp_matched[i] and temp_scores[i] >= det_thresh:
				temp.id = track_id
				temp.score = temp_scores[i]
				temp.scores += [[jj, temp_scores[i]]]
				track_id += 1
				obj_list += [temp]

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
			boxes[i,:] = obj_list_fr[i].bbox
			masks[:,:,i] = obj_list_fr[i].mask

		# save current frame with found objects
		# save_instances(image, boxes, masks, [x.class_name for x in obj_list_fr], 
		# 				class_names, ids = [str(x.id)[:4]+' '+'{:.2f}'.format(x.score) for x in obj_list_fr], 
		# 				file_name = str(jj)+'.png',colors=[x.color for x in obj_list_fr])
		# save_instances(image, boxes, masks, [-1 for x in obj_list_fr], 
		# 				class_names, ids = [str(x.id)[:4] for x in obj_list_fr], 
		# 				file_name = str(jj)+'.png',colors=[x.color for x in obj_list_fr])
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
	input_dir =  '/home/anthony/maskrcnn/Mask_RCNN/datasets/training/image_02/0017'
	pickle_dir = '/home/anthony/maskrcnn/Mask_RCNN/samples/pickles/0017s'
	# input_dir =  '/home/anthony/mbappe'
	# pickle_dir = '/home/anthony/maskrcnn/Mask_RCNN/samples/pickles/mbappe'
	# input_dir = '/home/anthony/nascar/frames'
	# pickle_dir = '/home/anthony/maskrcnn/Mask_RCNN/samples/pickles/nascar'
	from datetime import datetime 
	st_mot = datetime.now()
	y = [x.encoding for x in demo_mot(input_dir, pickle_dir ,use_extra_boxes = False)]
	print('\n\n\n ++++++++++++++++++\nMOT time: {}'.format(datetime.now()-st_mot))

	input_dir =  '/home/anthony/maskrcnn/Mask_RCNN/datasets/training/image_02/0014'
	pickle_dir = '/home/anthony/maskrcnn/Mask_RCNN/samples/pickles/0014s'

	st_mot = datetime.now()
	y = [x.encoding for x in demo_mot(input_dir, pickle_dir ,use_extra_boxes = False)]
	print('\n\n\n ++++++++++++++++++\nMOT time: {}'.format(datetime.now()-st_mot))
