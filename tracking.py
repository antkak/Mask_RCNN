import skimage.io
import os, sys
from os.path import isfile, join
from os import listdir
import uuid

ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

# Import measurement for tracking 
from measurements import save_instances, save_statistics





class trackedObject():

	def __init__(self, ID, mask, bbox, class_name, encodings, color = [1,1,1]):
		self.id = ID 
		self.mask = mask
		self.bbox = bbox
		self.class_name = class_name
		self.tracking_state = 'New'
		self.encoding = encodings
		self.color = color
		self._occluded_cnt = 0

	def refress_state(self, matched):
		if self.tracking_state == 'New':
			if matched: 
				self.tracking_state = 'Tracked'
			else:
				self.tracking_state = 'Occluded'
				self._occluded_cnt = 1
		elif self.tracking_state == 'Tracked':
			if not matched:
				self.tracking_state = 'Occluded'
				self._occluded_cnt = 1
		elif self.tracking_state == 'Occluded':
			if matched:
				self.tracking_state = 'Tracked'
				self._occluded_cnt = 0
			else:
				self._occluded_cnt += 1
				if self._occluded_cnt > 5: # Frames being occluded
					self.tracking_state = 'Lost'


def demo_mot(input_dir):

	##########################################################################
	################### initialize detection model ###########################
	##########################################################################

	# Directory to save logs and trained model
	MODEL_DIR = os.path.join(ROOT_DIR, "logs")

	# Local path to trained weights file
	COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")


	class InferenceConfig(coco.CocoConfig):
		# Set batch size to 1 since we'll be running inference on
		# one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
		GPU_COUNT = 1
		IMAGES_PER_GPU = 1

	config = InferenceConfig()
	config.display()

	# Create model object in inference mode.
	model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

	# Load weights trained on MS-COCO
	model.load_weights(COCO_MODEL_PATH, by_name=True)

	# COCO Class names
	# Index of the class in the list is its ID. For example, to get ID of
	# the teddy bear class, use: class_names.index('teddy bear')
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



	##########################################################################
	################### 		initialize input		######################
	########################################################################## 

	IMAGE_DIR = input_dir
	frames = sorted([f for f in listdir(IMAGE_DIR) if isfile(join(IMAGE_DIR, f))])

	# read first image
	image = skimage.io.imread(os.path.join(IMAGE_DIR,frames[0]))

	# run detection
	results = model.detect([image], verbose=1)

	# get more probable results
	r = results[0]
	save_instances(image, r['rois'], r['masks'], r['class_ids'], 
								class_names, r['scores'], file_name = 'temp.png')

	# obj_list = []
	# # for each object initialize trackedObject
	# for roi,mask,class_id in zip(r['rois'], r['masks'], r['class_ids']):
	# 	# find ID, mask, bbox, class_id, t color
	# 	obj_list += [trackedObject(uuid.uuid4(), mask, roi, class_id, None)]
	layers = model.get_trainable_layers()
	for layer in layers:
		if layer.name.startswith('fpn'):
			print("layer {0}\n\t output shape {1}".format(layer.name,layer.output.dims))
	assert(False)

		# from detector read appearance encoding (from correct pyramid layer)
		# find a way to initialize flow encoding
		# save trackedObject s in a list or something

	# for each following frame
		# read frame
		# Add extra anchors from previous step
		# GATING?
		# Run detection
		# for each newly found object initialize trackedObject
			# find ID, mask, bbox, class_name, tracked_id = New, color
			# from detector read appearance encoding (from correct pyramid layer)
			# run flownet and get motion encoding (is there a pyramid structure)
			# for each old object run simple score and get score matrix
		# run assingment 
		# for each newly found object change ID, color
		# for each tracked object change tracked ID
		# remove lost from active list

	return obj_list




def simple_score(object1, object2):
	# score object1.encoding, object2.encoding
	# multiple encodings, multiple regressing layers
	return 0

def assignment(score_matrix):
	# https://docs.scipy.org/doc/scipy-0.19.1/reference/generated/scipy.optimize.linear_sum_assignment.html
	# and thresholding on assignment score for None class??? better solution???
	return 0


if __name__ == '__main__':
	# # tests for refress_state method
	# import random
	# obj = trackedObject(0,0,0,0,0)
	# obj.tracking_state = 'New'
	# print(obj.tracking_state)
	# for i in range(100):
	# 	m = random.choice([True, False])
	# 	obj.refress_state(m)
	# 	print(str(obj.tracking_state)+'\t\t'+str(m))
	# tests for simple mot
	input_dir =  '/home/anthony/maskrcnn/Mask_RCNN/datasets/testing/image_02/0000'
	print([x.__dict__ for x in demo_mot(input_dir)])





