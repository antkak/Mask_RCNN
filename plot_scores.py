from matplotlib import pyplot as plt
import pickle


f = open('objects.pickle','rb')
r = pickle.load(f)

plt.figure(0)
i = 1
for obj in r:
	x = [x[0] for x in obj.scores]
	y = [x[1] for x in obj.scores]
	plt.figure(i)
	print(obj.id)
	plt.plot(x,y)
	plt.show()
	i+=1