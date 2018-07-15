import numpy as np
from scipy.spatial.distance import correlation, cosine, euclidean
from numba import jit
import cv2


def squarify(M,val):
	'''
	Pad matrix M with constant values val (right columns or below rows) so that is square
	'''
	(a,b)=M.shape
	if a>b:
		padding=((0,0),(0,a-b))
	else:
		padding=((0,b-a),(0,0))
	return np.pad(M,padding,mode='constant',constant_values=val)


def keepClasses(r, classes, class_names):
	'''Keep only relevant classes in detection step
	Detector can be trained to a superset of relevant classes
	such as car, pedestrian etc'''

	# Extract relevant class_indices
	class_indices = [class_names.index(c) for c in classes]

	# Get all indices of relevant classes
	# ri = [ i for i in range(len(r['class_ids'])) if (r['class_ids'][i] in class_indices and r['rois'][i,:][2]-r['rois'][i,:][0] >= 25 )]
	ri = [ i for i in range(len(r['class_ids'])) if r['class_ids'][i] in class_indices]

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


def simple_dist(object1, object2):
	'''
	find distance between two object encodings
	'''

	return cosine(object1.encoding,object2.encoding)


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

def tensor2vec(tensor, mode = 'full'):
	'''
	reaarange a tensor into a 1D vector
	if mode == 'svd' perform svd on the slices of the tensor
	and rearrange all singular values into a 1D vector
	(more computationally intensive but 1D vector now has length 
	M*D instead M*M*D ) 
	'''
	if mode == 'svd':
		s = np.linalg.svd(tensor, full_matrices=False, compute_uv=False)
		return np.transpose(s).flatten('F')
	elif mode == 'full':
		return tensor.flatten('F')

def pyr_sizes(level):
	'''
	Assign sample bbox width (= height) for current and lower pyramid level
	'''
	
	if level == 0:
		return 14, 10
	elif level == 1:
		return 28, 14
	elif level == 2:
		return 56, 28
	elif level == 3:
		return 112, 56
	elif level == 4:
		return 224, 112
	else:
		return 448, 224

def num_particles(mask):
	'''
	Return the number of particles as a function of mask dimensions
	Here I tried the square root + constant
	'''
	N = int(np.sqrt(np.sum(mask)))+10
	# N = np.sum(mask)//4
	# print('mask points:{} particles: {}'.format( np.sum(mask), N ) )
	return N//4, N


@jit(parallel=True)
def batch_cosine_dist(p1,p2):
	'''
	Compute the pairwise cosine distance of the rows of p1
	and p2
	'''
	l_w = 2
	p_len  = len(p1[0])
	p_card = len(p1)
	dist_matrix = np.zeros((p_card,p_card))
	for i in range(p_card):
		for j in range(p_card):
			dot = 0
			denom_a = 0
			denom_b = 0
			for k in range(p_len):
				dot += p1[i,k]*p2[j,k]
				denom_a += p1[i,k]*p1[i,k]
				denom_b += p2[j,k]*p2[j,k]
			dist_matrix[i,j] = 1 - dot/(np.sqrt(denom_a)*np.sqrt(denom_b))
	return dist_matrix


def bbs(obj1, obj2):
	'''
	A growing in efficiency implementation of best buddies similarity metric
	'''
	particles1 = obj1.encoding[1]
	boxes1 	  = obj1.encoding[0]
	particles2 = obj2.encoding[1]
	boxes2 	  = obj2.encoding[0]


	## Sample down the largest set of particles for unbiased metric
	if len(particles1) > len(particles2):
		# resample particles 1 in size of particles 2
		indices = np.random.choice(np.array(range(len(particles1))), len(particles2), replace=False) # should introduce probabilities? not sure
		particles1 = particles1[indices]
		boxes1 = boxes1[indices]
	else:
		# resample particles 2 in size of particles 1
		indices = np.random.choice(np.array(range(len(particles2))), len(particles1), replace=False) # should introduce probabilities? not sure
		particles2 = particles2[indices]
		boxes2 = boxes2[indices]


	# print(particles1.shape)
	# print(boxes1.shape)

	# print(particles2.shape)
	# print(boxes2.shape)
	# assert(False)

	# normalize coords dividing with the largest bounding boxes' dimensions (for occlusions) ? TODO?
	
	# include spatial distance for faithful bbs
	buddy = batch_cosine_dist(particles1, particles2)
	# print(buddy[0])
	# l_w = 2
	# for i in range(len(particles1)):
	# 	for j in range(len(particles2)):
	# 		buddy[i,j] = cosine(particles1[i], particles2[j]) #+ l_w*box_dist(boxes1[i], boxes2[j])

	# print(buddy[0])
	# count buddies
	# I found a marginally faster way 
	# and a CUDA one
	# A buddy is defined as an entry in a 2D matrix that is larger than all the other entries
	# in each row and column 
	buddies1 = np.zeros((len(particles1)))
	buddies2 = np.zeros((len(particles2)))
	for i in range(len(particles1)):
		buddies1[i] = np.argmin(buddy[i,:])
	for i in range(len(particles2)):
		buddies2[i] = np.argmin(buddy[:,i])

	buddy_count = 0
	for i in range(len(particles1)):
		index = buddies1[i]
		if buddies2[int(index)] == i:
			buddy_count += 1

	return buddy_count/min(len(particles1),len(particles2))

def random_sampling(mask, num_points):

	N = np.sum(mask)
	point_val = np.random.choice(np.array(list(range(1,N))), size = num_points, replace = False)

	return _random_sampling(mask,point_val)

@jit
def _random_sampling(mask, point_val):

	points_found = 0
	points = []
	# give id to each of the track
	ii = 1
	mask = mask.astype(int)
	for i in range(mask.shape[0]):
		for j in range(mask.shape[1]):
			if mask[i,j]==1:
				mask[i,j]= ii
				ii += 1

	points = []
	for val in point_val:
		inds = []
		for x in np.where(mask==val):
			inds += [np.asscalar(x)]
		points += [inds]
	return points

def sample_boxes(r, image=None):
	'''
	Samples boxes inside detection masks. 
	Returns:
	pyr_levels: List containing the computed pyramid level for each detection
	bboxes2_batch: A 2D numpy array [boxes][coordinates]
	split_list: List describing how to split bboxes2_batches  
	'''

	pyr_levels = []
	bboxes2_batch = np.array([0,0,0,0])
	split_list = []
	pyramid_erosion = -1
	# kernel = np.array([[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]] ,np.uint8)

	for i in range(len(r['class_ids'])):
		
		# feature pyramid that corresponds to object size
		pyramid_level = int(np.floor(4+np.log2(1/224*np.sqrt(np.abs((r['rois'][i,2]-r['rois'][i,0])*(r['rois'][i,3]-r['rois'][i,1]))))))
		# save to list to initialize object later
		pyr_levels += [pyramid_level]
		

		M1, M2 = pyr_sizes(pyramid_level) 
		M2 = M2//2
		kernel = np.ones(M2)

		if pyramid_level > pyramid_erosion:
			mask_image = cv2.erode((r['masks'][:,:,i]).astype(np.uint8), kernel, iterations=1).astype(bool)
		else:
			mask_image = r['masks'][:,:,i]

		# feature pyramid particle constants that correspond to object size
		# 1: same pyramid
		# 2: lower pyramid

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
		bboxes2_batch = np.vstack((bboxes2_batch,bboxes2))
		split_list += [N2]

	#visualize bounding boxes
	if image is not None:
	# 	for i in range(bboxes2_batch.shape[0]):
	# 		cv2.circle(image, ((bboxes2_batch[i,1]+bboxes2_batch[i,3])//2, (bboxes2_batch[i,0]+bboxes2_batch[i,2])//2), 1, (0,0,255), 1)
	# 	# for i in range(bboxes1.shape[0]):
	# 	# 	cv2.rectangle(image,tuple([bboxes1[i][1], bboxes1[i][0]]),tuple([bboxes1[i][3], bboxes1[i][2]]),(0,255,0),1)
		for i in list(np.random.choice(range(bboxes2_batch.shape[0]), 15)):
			cv2.rectangle(image,tuple([bboxes2_batch[i][1], bboxes2_batch[i][0]]),tuple([bboxes2_batch[i][3], bboxes2_batch[i][2]]),(0,255,0),1)

	# # normalize bboxes
	# # bboxes1 = np.array([normalize_boxes(bboxes1, image.shape[:2])])
	# 	cv2.imshow('im', image)
	# 	cv2.waitKey(0)

	return pyr_levels, bboxes2_batch, split_list, image
