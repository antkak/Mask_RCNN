import os
import sys
import logging
import random
import itertools
import colorsys

import numpy as np
from skimage.measure import find_contours
from matplotlib import pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import IPython.display
from os import listdir
from os.path import isfile, join
import skimage.io
from collections import Counter
np.seterr(divide='ignore', invalid='ignore')


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def random_colors(N, bright=True):
	"""
	Generate random colors.
	To get visually distinct colors, generate them in HSV space then
	convert to RGB.
	"""
	brightness = 1.0 if bright else 0.7
	hsv = [(i / N, 1, brightness) for i in range(N)]
	colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
	# random.shuffle(colors)
	return colors

def save_instances(image, boxes, masks, class_ids, class_names,
					  ids=None, title="",
					  figsize=(16, 16), ax=None,
					  show_mask=True, show_bbox=True,
					  colors=None, captions=None, file_name=None):
	"""
	boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
	masks: [height, width, num_instances]
	class_ids: [num_instances]
	class_names: list of class names of the dataset
	ids: (optional) tracking unique identity for each box
	title: (optional) Figure title
	show_mask, show_bbox: To show masks and bounding boxes or not
	figsize: (optional) the size of the image
	colors: (optional) An array or colors to use with each object
	captions: (optional) A list of strings to use as captions for each object
	"""
	# Number of instances

	N = boxes.shape[0]
	if not N:
		print("\n*** No instances to display *** \n")
	# else:
	# 	assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

	# If no axis is passed, create one and automatically call show()
	auto_show = False
	if not ax:
		_, ax = plt.subplots(1, figsize=figsize)
		auto_show = True

	# Generate random colors
	colors = colors or random_colors(N)

	# Show area outside image boundaries.
	height, width = image.shape[:2]
	ax.set_ylim(height + 10, -10)
	ax.set_xlim(-10, width + 10)
	ax.axis('off')
	ax.set_title(title)

	masked_image = image.astype(np.uint32).copy()
	for i in range(N):
		color = colors[i]
		#color = [1,1,1]
		# Bounding box
		if not np.any(boxes[i]):
			# Skip this instance. Has no bbox. Likely lost in image cropping.
			continue
		y1, x1, y2, x2 = boxes[i]
		if show_bbox:
			p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
								alpha=0.7, linestyle="dashed",
								edgecolor=color, facecolor='none')
			ax.add_patch(p)

		# Label
		if not captions:
			class_id = class_ids[i]
			identity = ids[i] if ids is not None else None
			label = class_names[class_id]
			# x = random.randint(x1, (x1 + x2) // 2)
			caption = "{} {}".format(label, identity) if identity else label
		else:
			caption = captions[i]
		ax.text(x1, y1 + 8, caption,
				color='w', size=11, backgroundcolor="none")

		# Mask
		
		if show_mask:
			mask = masks[:, :, i]
			masked_image = apply_mask(masked_image, mask, color)


			# Mask Polygon
			# Pad to ensure proper polygons for masks that touch image edges.
			padded_mask = np.zeros(
				(mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
			padded_mask[1:-1, 1:-1] = mask
			contours = find_contours(padded_mask, 0.5)
			for verts in contours:
				# Subtract the padding and flip (y, x) to (x, y)
				verts = np.fliplr(verts) - 1
				p = Polygon(verts, facecolor="none", edgecolor=color)
				ax.add_patch(p)
	ax.imshow(masked_image.astype(np.uint8))
	plt.savefig(file_name)
	plt.close('all')
	return masked_image.astype(np.uint8)

def save_statistics(output_file, file_name, tracked_id, rois, class_names, class_ids, scores):

	# file_name to frame enumeration
	frame = int(file_name[0:6])

	# dictionary for class mapping

	with open(output_file,'a') as statfile:
		tid = tracked_id
		for track_id, roi, class_id, score in zip(range(len(rois)),rois,class_ids,scores):
			statfile.write('{0} {1} {2} 0 0 0 {3} {4} {5} {6} 0 0 0 0 0 0 0 {7}\n'.format(str(frame), 
				tid, class_names[class_id],roi[0],roi[1], roi[2], roi[3], str(score)))
			tid += 1

def track_len_histogram(gt_dir, file_name=None):

	with open(gt_dir,'r') as gt_file:
		gt_lines = [x.split(' ') for x in gt_file.readlines()]

	tracks = [x[1] for x in gt_lines]

	tr_counts = Counter(tracks)
	height = tr_counts.values()
	x = tr_counts.keys()
	plt.bar(x,height)
	if not file_name:
		plt.show()
	else:
		plt.savefig(file_name)
		plt.close('all')
	

def visualize_gt(frames_dir, gt_dir, save_dir):


	class_names = ['Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc', 'DontCare' ]

	# open gt_dir/gt.txt and save data
	with open(gt_dir,'r') as gt_file:
		gt_lines = [x.split(' ') for x in gt_file.readlines()]
	
	# Load image frames
	file_names = [f for f in listdir(frames_dir) if isfile(join(frames_dir, f))]
	
	# exract data from file and save image with gt
	for frame, file_name in enumerate(sorted(file_names)):

		# read frame
		image = skimage.io.imread(os.path.join(frames_dir,file_name))
		
		# extract relevant lines for frame
		lines = [x for x in gt_lines if x[0]==str(frame)]

		# extract relevant data for frame
		boxes = np.array([list(map(np.floor,list(map(float,[x[7],x[6],x[9],x[8]])))) for x in lines]) # x1,y1,x2,y2 to y1, x1, y2, x2 
																									  # cast to float from string
																									  # floor to get int coords
																									  # convert to np array
		scores = np.array(list(map(float,[x[-1] for x in lines])))
		class_ids = np.array([class_names.index(x[2]) for x in lines]) 

		# save frame with gt
		save_instances(image, boxes, np.array([0]*len(boxes)), class_ids, class_names,
						  ids=scores, ax=None, colors=[[1,1,1]]*len(boxes),
						  show_mask=False, show_bbox=True,
						  file_name=os.path.join(save_dir,str(frame)+'.png'))

def plot_tracks(frames_dir, gt_dir):
	import cv2
	# open gt_dir/gt.txt and save data
	with open(gt_dir,'r') as gt_file:
		gt_lines = [x.split(' ') for x in gt_file.readlines()]
	
	# Load image frames
	file_names = [f for f in listdir(frames_dir) if isfile(join(frames_dir, f))]

	# read frame
	image = skimage.io.imread(os.path.join(frames_dir,sorted(file_names)[0]))
	
	# exract data from file and save image with gt
	boxes = []
	tracks = []
	tid = str(0)
	for frame, file_name in enumerate(sorted(file_names)):
		
		# extract relevant lines for frame
		lines = [x for x in gt_lines if x[0]==str(frame)]

		# extract relevant data for frame
		boxes += [list(map(int,list(map(float,[x[7],x[6],x[9],x[8]])))) for x in lines if x[1] == tid ] # x1,y1,x2,y2 to y1, x1, y2, x2 
																									  # cast to float from string
																									  # floor to get int coords
																									  # convert to np array
		tracks += [int(x[1]) for x in lines if x[1] == tid]	

	colors = random_colors(len(tracks))					


	for i, box in enumerate(boxes):																							  
		cv2.circle(image, ((box[1]+box[3])//2, (box[0]+box[2])//2), 1, tuple(255*np.array(colors[i])), 1)

	cv2.imshow('im',image)
	cv2.waitKey(0)

def curv(a,th):
	if len(a) < th:
		return 0
	a = np.array(a[-th:])
	print(a)
	dx_dt = np.gradient(a[:, 0])
	dy_dt = np.gradient(a[:, 1])
	ds_dt = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)
	d2x_dt2 = np.gradient(dx_dt)
	d2y_dt2 = np.gradient(dy_dt)
	curvature =   ds_dt*(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5
	#print(curvature)
	return curvature[-3]


def compute_curvatures(frames_dir, gt_dir):
	import cv2
	# open gt_dir/gt.txt and save data
	with open(gt_dir,'r') as gt_file:
		gt_lines = [x.split(' ') for x in gt_file.readlines()]
	
	# Load image frames
	file_names = [f for f in listdir(frames_dir) if isfile(join(frames_dir, f))]

	# read frame
	image = skimage.io.imread(os.path.join(frames_dir,sorted(file_names)[0]))

	objects = list(set([x[1] for x in gt_lines ]))
	obj_num = len(list(set([x[1] for x in gt_lines ]))) - 1
	leg = []
	
	# exract data from file and save image with gt
	for tr_i in range(obj_num):
		print(tr_i)
		boxes = []
		track_s = 0
		tid = str(tr_i)
		curvature = []
		box_t = []
		start_track  = False


		for frame, file_name in enumerate(sorted(file_names)):
			
			# extract relevant lines for frame
			lines = [x for x in gt_lines if x[0]==str(frame)]


			# extract relevant data for frame
			box = [list(map(int,list(map(float,[x[7],x[6],x[9],x[8]])))) for x in lines if x[1] == tid ] # x1,y1,x2,y2 to y1, x1, y2, x2 
																										  # cast to float from string
																										  # floor to get int coords
																										  # convert to np array

			if len(box) > 0:
				box = box[0]
				start_track = True
				boxes += [[(box[1]+box[3])//2, (box[0]+box[2])//2]]
				track_s += 1	
				thresh = 5
				curvature += [curv(boxes, thresh)]

			else:
				if not start_track:
					curvature += [0]
				else:
					if len(boxes) > 2:
						db = [boxes[-1][0] - boxes[-2][0], boxes[-1][1] - boxes[-2][1]]
						boxes += [[boxes[-1][0] + db[0], boxes[-1][1] + db[1]]]
						curvature += [curv(boxes, thresh)]
					else:
						boxes += [boxes[-1]]
						curvature += [curvature[-1]]

		colors = random_colors(track_s)					
		# if tid == '1':
		# 	for i, box in enumerate(boxes):																							  
		# 		cv2.circle(image, (box[0], box[1]), 1, tuple(255*np.array(colors[i])), 1)
		plt.plot(curvature)
		leg += [tr_i]


	plt.legend(leg, loc='upper left')
	plt.show()

	cv2.imshow('im',image)
	cv2.waitKey(0)		


if __name__ == "__main__":

	frames_dir = '/home/anthony/maskrcnn/Mask_RCNN/datasets/training/image_02/0014'
	# gt_dir = '/home/anthony/maskrcnn/Mask_RCNN/datasets/kitti_track/training/label_02/0017.txt'
	gt_dir = '/home/anthony/maskrcnn/Mask_RCNN/0014.txt'
	save_dir = '/home/anthony/maskrcnn/Mask_RCNN/samples/temp'
	# visualize_gt(frames_dir,gt_dir,save_dir)
	# plot_tracks(frames_dir, gt_dir)
	compute_curvatures(frames_dir, gt_dir)

# 	gt_names = [f for f in listdir(gt_dir) if isfile(join(gt_dir, f))]
# 	for name in gt_names:
# 		track_len_histogram(join(gt_dir,name), file_name = name[:-3]+'png')