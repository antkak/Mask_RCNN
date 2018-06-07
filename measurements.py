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
	random.shuffle(colors)
	return colors

def save_instances(image, boxes, masks, class_ids, class_names,
					  scores=None, title="",
					  figsize=(16, 16), ax=None,
					  show_mask=True, show_bbox=True,
					  colors=None, captions=None, file_name=None):
	"""
	boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
	masks: [height, width, num_instances]
	class_ids: [num_instances]
	class_names: list of class names of the dataset
	scores: (optional) confidence scores for each box
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
	else:
		assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

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
			score = scores[i] if scores is not None else None
			label = class_names[class_id]
			x = random.randint(x1, (x1 + x2) // 2)
			caption = "{} {:.3f}".format(label, score) if score else label
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
		scores = np.array(list(map(float,[x[-1] for x in lines])))
		class_ids = np.array([class_names.index(x[2]) for x in lines]) 

		# save frame with gt
		save_instances(image, boxes, np.array([0]*len(boxes)), class_ids, class_names,
						  scores=scores, ax=None, colors=[[1,1,1]]*len(boxes),
						  show_mask=False, show_bbox=True,
						  file_name=os.path.join(save_dir,str(frame)+'.png'))

if __name__ == "__main__":

	frames_dir = '/home/anthony/maskrcnn/Mask_RCNN/datasets/training/image_02/0000'
	gt_dir = '/home/anthony/maskrcnn/Mask_RCNN/datasets/kitti_track/training/label_02/0000.txt'
	save_dir = '/home/anthony/maskrcnn/Mask_RCNN/samples/temp'
	visualize_gt(frames_dir,gt_dir,save_dir)
