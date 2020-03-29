import numpy as np
import miniBatchGD as mBGD


def LoadBatch(filename):
    """ Copied from the dataset website """
    from scipy.io import loadmat
    with open('datasets/cifar-10-batches-mat/'+filename, 'rb') as fo:
        dict = loadmat(fo)
        X = np.transpose(dict['data'])
        y = np.transpose(dict['labels'])
        Y = np.zeros([10,X.shape[1]])
        for i in range(X.shape[1]):
            Y[y[0,i],i] = 1
    return [X,Y,y]

def ComputeGradsNum(X, Y, W, b, lamda, h):
	""" Converted from matlab code """
	no 	= 	W.shape[0]
	d 	= 	X.shape[0]

	grad_W = np.zeros(W.shape);
	grad_b = np.zeros((no, 1));

	test_num = mBGD.MiniBatchGD(X, Y, [], [0, 0, 0, lamda], W, b);test_num.Softmax();[l,c] = test_num.ComputeCost()
    
    #c = ComputeCost(X, Y, W, b, lamda)
	
    
	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] += h;test_num2 = mBGD.MiniBatchGD(X, Y, [], [0, 0, 0, lamda], W, b_try);test_num2.Softmax();[l2,c2] = test_num2.ComputeCost()
#		c2 = ComputeCost(X, Y, W, b_try, lamda)
		grad_b[i] = (c2-c) / h

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i,j] += h;test_num2 = mBGD.MiniBatchGD(X, Y, [], [0, 0, 0, lamda], W_try, b);test_num2.Softmax();[l2,c2] = test_num2.ComputeCost()
#			c2 = ComputeCost(X, Y, W_try, b, lamda)
			grad_W[i,j] = (c2-c) / h

	return [grad_W, grad_b]

def ComputeGradsNumSlow(X, Y, W, b, lamda, h):
	""" Converted from matlab code """
	no 	= 	W.shape[0]
	d 	= 	X.shape[0]

	grad_W = np.zeros(W.shape);
	grad_b = np.zeros((no, 1));
	
	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] -= h;test_num1 = mBGD.MiniBatchGD(X, Y, [], [0, 0, 0, lamda], W, b_try);test_num1.Softmax();[l1,c1] = test_num1.ComputeCost()
#		c1 = ComputeCost(X, Y, W, b_try, lamda)

		b_try = np.array(b)
		b_try[i] += h;test_num2 = mBGD.MiniBatchGD(X, Y, [], [0, 0, 0, lamda], W, b_try);test_num2.Softmax();[l2,c2] = test_num2.ComputeCost()
#		c2 = ComputeCost(X, Y, W, b_try, lamda)

		grad_b[i] = (c2-c1) / (2*h)

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i,j] -= h;test_num1 = mBGD.MiniBatchGD(X, Y, [], [0, 0, 0, lamda], W_try, b);test_num1.Softmax();[l1,c1] = test_num1.ComputeCost()
#			c1 = ComputeCost(X, Y, W_try, b, lamda)

			W_try = np.array(W)
			W_try[i,j] += h;test_num2 = mBGD.MiniBatchGD(X, Y, [], [0, 0, 0, lamda], W_try, b);test_num2.Softmax();[l2,c2] = test_num2.ComputeCost()
#			c2 = ComputeCost(X, Y, W_try, b, lamda)
#
			grad_W[i,j] = (c2-c1) / (2*h)

	return [grad_W, grad_b]

def montage(typ,W,range1,range2, param, name):
    """ Display the image for each label in W """
    import matplotlib.pyplot as plt
    if typ == "photo":
        fig, ax = plt.subplots(range1,range2)
        for i in range(range1):
            for j in range(range2):   
                im  = W[:,5*i+j].reshape(32,32,3, order='F')
                sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
                sim = sim.transpose(1,0,2)
                ax[i][j].imshow(sim, interpolation='nearest')
                ax[i][j].set_title("y="+str(5*i+j))
                ax[i][j].axis('off')
    if typ == "weight":        
        fig, ax = plt.subplots(range2,range1)
        fig.canvas.set_window_title(name + ' - Weight matrix: parameter-set ' + str(param))
        for i in range(range1):
            im  = W[i,:].reshape(32,32,3, order='F')
            sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
            sim = sim.transpose(1,0,2)
            ax[i].imshow(sim, interpolation='nearest')
            ax[i].set_title("y="+str(i))
            ax[i].axis('off')
    plt.show()

def save_as_mat(data, name="model"):
	""" Used to transfer a python model to matlab """
	import scipy.io as sio
	sio.savemat(name,'.mat',{name:b})