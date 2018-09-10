import os
# Root directory of the project
ROOT_DIR = os.path.abspath("../")
from mrcnn import utils

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

if not os.path.exists(COCO_MODEL_PATH):
	utils.download_trained_weights(COCO_MODEL_PATH)
