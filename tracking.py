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


def simple_score(object1, object2):
	# score object1.encoding, object2.encoding
	# multiple encodings, multiple regressing layers
	return 0

def assignment(score_matrix):
	# https://docs.scipy.org/doc/scipy-0.19.1/reference/generated/scipy.optimize.linear_sum_assignment.html
	# and thresholding on assignment score for None class??? better solution???
	return 0


if __name__ == '__main__':
	# tests for refress_state method
	import random
	obj = trackedObject(0,0,0,0,0)
	obj.tracking_state = 'New'
	print(obj.tracking_state)
	for i in range(100):
		m = random.choice([True, False])
		obj.refress_state(m)
		print(str(obj.tracking_state)+'\t\t'+str(m))




