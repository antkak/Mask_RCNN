import numpy as np
from numba import jit
import cv2
import skimage
from scipy.spatial.distance import mahalanobis
np.random.seed(1)
import scipy.stats as st

# [x]
def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel


def gating(x_1, P_1, x_2, b_gain,s = 3):
   
    lambda_, v = np.linalg.eig(P_1[:2,:2])
    lambda_ = np.sqrt(lambda_)

    ll_p   = (s*lambda_[0]+b_gain)**2*np.linalg.multi_dot([v[0].T, np.linalg.inv(P_1[:2,:2]), v[0]])
    ll_x_2 = mahalanobis(x_2[:2],x_1[:2],np.linalg.inv(P_1[:2,:2]))**2

    if ll_x_2 >= ll_p:
        return True
    else:
        return False

# [x]
def gating_mask(x_1, P_1, imshape, s = 3):
    
    lambda_, v = np.linalg.eig(P_1[:2,:2])
    lambda_ = np.sqrt(lambda_)
    ll_p   = (s*lambda_[0])**2*np.linalg.multi_dot([v[0].T, np.linalg.inv(P_1[:2,:2]), v[0]])

    print(imshape)
    ecl_mask = np.zeros(tuple(imshape))
    for i in range(imshape[0]):
    	for j in range(imshape[1]):
    		x = np.array([i,j])
    		if mahalanobis(x,x_1[:2],np.linalg.inv(P_1[:2,:2]))**2 <= ll_p:
    			ecl_mask[i,j] = 1
    return ecl_mask



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
	ri = [ i for i in range(len(r['class_ids'])) if (r['class_ids'][i] in class_indices and \
			r['rois'][i,:][2]-r['rois'][i,:][0] >= 25 )]
	# ri = [ i for i in range(len(r['class_ids'])) if r['class_ids'][i] in class_indices]

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
	return int(np.floor(7*np.exp2(level+1))), int(np.floor(7*np.exp2(level)))

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
	#  https://nyu-cds.github.io/python-numba/05-cuda/
	'''
	Compute the pairwise cosine distance of the rows of p1
	and p2. For non normalized vectors
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
			dist_matrix[i,j] = 1 - dot
	return dist_matrix


def bbs(obj1, obj2, out_buddies=False):
	'''
	A growing in efficiency implementation of best buddies similarity metric
	returns: bbs metric
			 bb pairs (boxes)
	'''
	particles1 = obj1.encoding[1]
	boxes1 	   = obj1.encoding[0]
	probs1     = obj1.encoding[2]
	particles2 = obj2.encoding[1]
	boxes2 	   = obj2.encoding[0]
	probs2     = obj2.encoding[2]


	## Sample down the largest set of particles for unbiased metric
	if len(particles1) > len(particles2):
		# resample particles 1 in size of particles 2
		l2 = False
		indices = np.random.choice(np.array(range(len(particles1))), len(particles2), replace=False) # should introduce probabilities? not sure
		particles1 = particles1[indices]
		boxes1 = boxes1[indices]

	else:
		l2 = True
		# resample particles 2 in size of particles 1
		indices = np.random.choice(np.array(range(len(particles2))), len(particles1), replace=False) # should introduce probabilities? not sure
		particles2 = particles2[indices]
		boxes2 = boxes2[indices]
	
	# for normalized vectors (super fast)
	buddy = 1 - np.dot(particles1, particles2.T)

	# count buddies
	# A buddy is defined as an entry in a 2D matrix that is larger than all the other entries
	# in each row and column 
	buddies1 = np.zeros((len(particles1)))
	buddies2 = np.zeros((len(particles2)))
	for i in range(len(particles1)):
		buddies1[i] = np.argmin(buddy[i,:])
	for i in range(len(particles2)):
		buddies2[i] = np.argmin(buddy[:,i])

	# count best buddies, if out_buddies return buddy coordinates 
	buddy_b = []
	if out_buddies:
		buddy_count = 0
		for i in range(len(particles1)):
			index = buddies1[i]
			if buddies2[int(index)] == i:
				buddy_count += 1
				buddy_b += [[list(boxes1[i]),list(boxes2[int(index)])]]
	else:
		buddy_count = 0
		for i in range(len(particles1)):
			index = buddies1[i]
			if buddies2[int(index)] == i:
				buddy_count += 1

	return buddy_count/min(len(particles1),len(particles2)), buddy_b

def random_sampling(mask, num_points):
	'''
	Samples randomly %num_points points (x,y) inside the mask image 
	'''

	N = np.sum(mask)
	if num_points > N:
		num_points = N

	pairs = np.array(list(zip(*np.where(mask == 1))))
	point_val = np.random.choice(np.array(list(range(N))), size = num_points, replace = False)

	return pairs[point_val]


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

	for i in range(len(r['class_ids'])):
		
		# feature pyramid that corresponds to object size
		pyr = 4+np.log2(1/224*np.sqrt(np.abs((r['rois'][i,2]-r['rois'][i,0])*(r['rois'][i,3]-r['rois'][i,1]))))
		pyramid_level = int(np.floor(pyr))
		# save to list to initialize object later
		pyr_levels += [pyramid_level]
		

		M1, M2 = pyr_sizes(pyr) 
		M2 = M2//4

		kernel = skimage.morphology.diamond(M2//2)

		if pyramid_level > pyramid_erosion:
			mask_image = cv2.erode((r['masks'][:,:,i]).astype(np.uint8), kernel, iterations=1).astype(bool)
		else:
			mask_image = r['masks'][:,:,i]

		# feature pyramid particle constants that correspond to object size
		N1, N2 = num_particles(mask_image)
		N2 *= 4

		# visualize particle constants
		# cv2.circle(image, ((r['rois'][i,1]+r['rois'][i,3])//2, (r['rois'][i,0]+r['rois'][i,2])//2), M2//2, (0,0,255), 1)

		# sample points inside mask
		points2 = random_sampling(mask_image, N2)

		M1 = M1//2
		M2 = M2//2

		# points to bounding boxes
		bboxes2 = np.array([[ point[0]-M2, point[1]-M2, point[0]+M2, point[1]+M2 ] for point in points2])
		bboxes2_batch = np.vstack((bboxes2_batch,bboxes2))
		split_list += [N2]

	#visualize bounding boxes
	if image is not None:
		for i in range(bboxes2_batch.shape[0]):
			cv2.circle(image, ((bboxes2_batch[i,1]+bboxes2_batch[i,3])//2, (bboxes2_batch[i,0]+bboxes2_batch[i,2])//2), 1, (0,0,255), 1)
		for i in list(np.random.choice(range(bboxes2_batch.shape[0]), 15)):
			cv2.rectangle(image,tuple([bboxes2_batch[i][1], bboxes2_batch[i][0]]),tuple([bboxes2_batch[i][3], bboxes2_batch[i][2]]),(0,255,0),1)

	# 	cv2.imshow('im', image)
	# 	cv2.waitKey(0)
	else:
		image = None

	return pyr_levels, bboxes2_batch, split_list, image

def best_buddies_assignment(peek_matrix):

	buddies1 = np.zeros((peek_matrix.shape[0]))
	buddies2 = np.zeros((peek_matrix.shape[1]))
	for i in range(peek_matrix.shape[0]):
		buddies1[i] = np.argmin(peek_matrix[i,:])
	for i in range(peek_matrix.shape[1]):
		buddies2[i] = np.argmin(peek_matrix[:,i])

	buddy_count = 0
	row_ind = []
	col_ind = []
	rows = list(range(peek_matrix.shape[0]))
	cols = list(range(peek_matrix.shape[1]))
	for i in range(peek_matrix.shape[0]):
		index = buddies1[i]
		if buddies2[int(index)] == i:
			buddy_count += 1
			row_ind += [i]
			rows.remove(i)
			col_ind += [int(index)]
			cols.remove(int(index))
			# buddy_b += [[list(boxes1[i]),list(boxes2[int(index)])]]	

	print(row_ind)
	print(col_ind)


	if len(rows) > 0:

		# get cost submatrix
		peek_matrix2 = peek_matrix[np.array(rows)[:,None],cols]
		print(peek_matrix2)
		# print(peek_matrix2)

		# solve the lin sum assignment
		if peek_matrix2.shape[0]==1:
			row_ind += [rows[0]]
			col_ind += [cols[np.argmin(peek_matrix2)]]
		else:
			row_left, col_left = linear_sum_assignment(peek_matrix2)
			row_ind += [rows[i] for i in row_left]
			col_ind += [cols[i] for i in col_left]

	row_ind = np.array(row_ind)
	col_ind = np.array(col_ind)
	inds = row_ind.argsort()
	row_ind = list(row_ind[inds])
	col_ind = list(col_ind[inds])

	return row_ind, col_ind
