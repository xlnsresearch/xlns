##############################################
# 
# used NN code from Arnab Sanyal USC. Spring 2019
# arn_generic.py <- arnnpb.py <- arnabs8xlnsnpv2024.py <- arnabs8xlnsnp2024r.py
# code that works with all classes and with FP 
###################################################

import numpy as np
import matplotlib.pyplot as plt
import argparse
import math
import xlns as xl
import os
import time

import xlnsconf.xlnsudFracnorm  #uncomment for Zhang/Han or Paliouras fractional normalize
xlnsconf.xlnsudFracnorm.ilog2 = xlnsconf.xlnsudFracnorm.ipallog2 #also uncomment only for Paliouras
xlnsconf.xlnsudFracnorm.ipow2 = xlnsconf.xlnsudFracnorm.ipalpow2

#import lpvip_ufunc
#xl.sbdb_ufunc = lpvip_ufunc.sbdb_ufunc_lpvip

xl.xlnssetF(10)

def softmax(inp):
	max_vals = inp.max(axis=1)
	max_vals = xl.reshape(max_vals,(xl.size(max_vals), 1))
	u = xl.exp(inp - max_vals)
	v = u.sum(axis=1)
	v = v.reshape((xl.size(v), 1))
	u = u / v
	return u

def main(main_params):
	print("arbitrary base np LNS. Also xl.hstack, xl. routines in softmax")
	print("testing new softmax and * instead of @ for delta")
	print("works with type "+main_params['type'])

	is_training = bool(main_params['is_training'])
	leaking_coeff = float(main_params['leaking_coeff'])
	batchsize = int(main_params['minibatch_size'])
	lr = float(main_params['learning_rate'])
	num_epoch = int(main_params['num_epoch'])
	_lambda = float(main_params['lambda'])
	ones = np.array((list(np.ones((batchsize, 1)))))
		
	if is_training:
		# load mnist data and split into train and test sets
		# one-hot encoded target column

		file = np.load('./mnist.npz', 'r') # dataset
		x_train = ((file['train_data']))
		y_train = np.array((file['train_labels']))
		x_test = np.array((file['test_data']))
		y_test = np.array((file['test_labels']))
		file.close()

		split = int(main_params['split'])
		x_val = x_train[split:]
		print('#=',split,' xlns b=','two ',' F=',xl.xlnsF,' B=',xl.xlnsB, ' batch=',batchsize, ' lr=',lr)
		y_val = y_train[split:]
		y_train = y_train[:split]
		x_train = x_train[:split]


		if os.path.isfile("./weightin.npz"):
		    print("using ./weightin.npz")
		    randfile = np.load("./weightin.npz","r")
		    W1 = randfile["W1"] 
		    W2 = randfile["W2"]
		    randfile.close()
		else:
		    print("using new random weights")
		    W1 = np.array((list(np.random.normal(0, 0.1, (785, 100)))))
		    W2 = np.array((list(np.random.normal(0, 0.1, (101, 10)))))
		    np.savez_compressed("./weightout.npz",W1=W1,W2=W2)
		delta_W1 = np.array((list(np.zeros(W1.shape))))
		delta_W2 = np.array((list(np.zeros(W2.shape))))

		if main_params['type'] == 'xlnsnp': 
		  lnsW1 = xl.xlnsnp(np.array(xl.xlnscopy(list(W1))))
		  lnsW2 = xl.xlnsnp(np.array(xl.xlnscopy(list(W2))))
		  lnsones = xl.xlnsnp(np.array(xl.xlnscopy(list(np.ones((batchsize, 1))))))
		  lnsdelta_W1 = xl.xlnsnp(np.array(xl.xlnscopy(list(np.zeros(W1.shape)))))
		  lnsdelta_W2 = xl.xlnsnp(np.array(xl.xlnscopy(list(np.zeros(W2.shape)))))
		if main_params['type'] == 'xlnsnpv': 
		  lnsW1 = xl.xlnsnpv(np.array(xl.xlnscopy(list(W1))),6)
		  lnsW2 = xl.xlnsnpv(np.array(xl.xlnscopy(list(W2))),6)
		  lnsones = xl.xlnsnpv(np.array(xl.xlnscopy(list(np.ones((batchsize, 1))))))
		  lnsdelta_W1 = xl.xlnsnpv(np.array(xl.xlnscopy(list(np.zeros(W1.shape)))))
		  lnsdelta_W2 = xl.xlnsnpv(np.array(xl.xlnscopy(list(np.zeros(W2.shape)))))
		if main_params['type'] == 'xlnsnpb': 
		  lnsW1 = xl.xlnsnpb(np.array(xl.xlnscopy(list(W1))),2**2**-6)
		  lnsW2 = xl.xlnsnpb(np.array(xl.xlnscopy(list(W2))),2**2**-6)
		  lnsones = xl.xlnsnpb(np.array(xl.xlnscopy(list(np.ones((batchsize, 1))))),2**2**-xl.xlnsF)
		  lnsdelta_W1 = xl.xlnsnpb(np.array(xl.xlnscopy(list(np.zeros(W1.shape)))),2**2**-xl.xlnsF)
		  lnsdelta_W2 = xl.xlnsnpb(np.array(xl.xlnscopy(list(np.zeros(W2.shape)))),2**2**-xl.xlnsF)
		if main_params['type'] == 'xlns': 
		  lnsW1 = (np.array(xl.xlnscopy(list(W1))))
		  lnsW2 = (np.array(xl.xlnscopy(list(W2))))
		  lnsones = (np.array(xl.xlnscopy(list(np.ones((batchsize, 1))))))
		  lnsdelta_W1 = (np.array(xl.xlnscopy(list(np.zeros(W1.shape)))))
		  lnsdelta_W2 = (np.array(xl.xlnscopy(list(np.zeros(W2.shape)))))
		if main_params['type'] == 'xlnsud': 
		  lnsW1 = (np.array(xl.xlnscopy(list(W1),xl.xlnsud)))
		  lnsW2 = (np.array(xl.xlnscopy(list(W2),xl.xlnsud)))
		  lnsones = (np.array(xl.xlnscopy(list(np.ones((batchsize, 1))),xl.xlnsud)))
		  lnsdelta_W1 = (np.array(xl.xlnscopy(list(np.zeros(W1.shape)),xl.xlnsud)))
		  lnsdelta_W2 = (np.array(xl.xlnscopy(list(np.zeros(W2.shape)),xl.xlnsud)))
		if main_params['type'] == 'xlnsv': 
		  lnsW1 = (np.array(xl.xlnscopy(list(W1),xl.xlnsv,6)))
		  lnsW2 = (np.array(xl.xlnscopy(list(W2),xl.xlnsv,6)))
		  lnsones = (np.array(xl.xlnscopy(list(np.ones((batchsize, 1))),xl.xlnsv)))
		  lnsdelta_W1 = (np.array(xl.xlnscopy(list(np.zeros(W1.shape)),xl.xlnsv)))
		  lnsdelta_W2 = (np.array(xl.xlnscopy(list(np.zeros(W2.shape)),xl.xlnsv)))
		if main_params['type'] == 'xlnsb': 
		  lnsW1 = (np.array(xl.xlnscopy(list(W1),xl.xlnsb,2**2**-6)))
		  lnsW2 = (np.array(xl.xlnscopy(list(W2),xl.xlnsb,2**2**-6)))
		  lnsones = (np.array(xl.xlnscopy(list(np.ones((batchsize, 1))),xl.xlnsb,2**2**-xl.xlnsF)))
		  lnsdelta_W1 = (np.array(xl.xlnscopy(list(np.zeros(W1.shape)),xl.xlnsb,2**2**-xl.xlnsF)))
		  lnsdelta_W2 = (np.array(xl.xlnscopy(list(np.zeros(W2.shape)),xl.xlnsb,2**2**-xl.xlnsF)))
		if main_params['type'] == 'float': 
		  lnsW1 = (np.array((list(W1))))
		  lnsW2 = (np.array((list(W2))))
		  lnsones = (np.array((list(np.ones((batchsize, 1))))))
		  lnsdelta_W1 = (np.array((list(np.zeros(W1.shape)))))
		  lnsdelta_W2 = (np.array((list(np.zeros(W2.shape)))))


		performance = {}
		performance['lnsacc_train'] = np.zeros(num_epoch)
		performance['lnsacc_val'] = np.zeros(num_epoch)
		start_time = time.process_time()

		for epoch in range(num_epoch):
			print('At Epoch %d:' % (1 + epoch))
			for mbatch in range(int(split / batchsize)):
				start = mbatch * batchsize
				x = np.array((list(x_train[start:(start + batchsize)])))
				y = np.array((list(y_train[start:(start + batchsize)])))
				if main_params['type'] == 'xlnsnp':
				  lnsx = xl.xlnsnp(np.array(xl.xlnscopy(np.array(x,dtype=np.float64))))
				  lnsy = xl.xlnsnp(np.array(xl.xlnscopy(np.array(y,dtype=np.float64))))
				if main_params['type'] == 'xlnsnpv':
				  lnsx = xl.xlnsnpv(np.array(xl.xlnscopy(np.array(x,dtype=np.float64))))
				  lnsy = xl.xlnsnpv(np.array(xl.xlnscopy(np.array(y,dtype=np.float64))))
				if main_params['type'] == 'xlnsnpb':
				  lnsx = xl.xlnsnpb(np.array(xl.xlnscopy(np.array(x,dtype=np.float64))),2**2**-xl.xlnsF)
				  lnsy = xl.xlnsnpb(np.array(xl.xlnscopy(np.array(y,dtype=np.float64))),2**2**-xl.xlnsF)
				if main_params['type'] == 'xlns':
				  lnsx = (np.array(xl.xlnscopy(np.array(x,dtype=np.float64))))
				  lnsy = (np.array(xl.xlnscopy(np.array(y,dtype=np.float64))))
				if main_params['type'] == 'xlnsud':
				  lnsx = (np.array(xl.xlnscopy(np.array(x,dtype=np.float64),xl.xlnsud)))
				  lnsy = (np.array(xl.xlnscopy(np.array(y,dtype=np.float64),xl.xlnsud)))
				if main_params['type'] == 'xlnsv':
				  lnsx = (np.array(xl.xlnscopy(np.array(x,dtype=np.float64),xl.xlnsv)))
				  lnsy = (np.array(xl.xlnscopy(np.array(y,dtype=np.float64),xl.xlnsv)))
				if main_params['type'] == 'xlnsb':
				  lnsx = (np.array(xl.xlnscopy(np.array(x,dtype=np.float64),xl.xlnsv,2**2**-xl.xlnsF)))
				  lnsy = (np.array(xl.xlnscopy(np.array(y,dtype=np.float64),xl.xlnsv,2**2**-xl.xlnsF)))
				if main_params['type'] == 'float':
				  lnsx = (np.array((np.array(x,dtype=np.float64))))
				  lnsy = (np.array((np.array(y,dtype=np.float64))))
				lnss1 = xl.hstack((lnsones, lnsx)) @ lnsW1
				lnsmask = (lnss1 > 0) + (leaking_coeff * (lnss1 < 0))
				lnsa1 = lnss1 * lnsmask
				lnss2 = xl.hstack((lnsones, lnsa1)) @ lnsW2
				lnsa2 = softmax(lnss2)
				lnsgrad_s2 = (lnsa2 - lnsy) / batchsize
				lnsgrad_a1 = lnsgrad_s2 @ xl.transpose(lnsW2[1:])
				lnsdelta_W2 = xl.transpose(xl.hstack((lnsones, lnsa1))) @ lnsgrad_s2
				lnsgrad_s1 = lnsmask * lnsgrad_a1
				lnsdelta_W1 = xl.transpose(xl.hstack((lnsones, lnsx))) @ lnsgrad_s1
				lnsW2 -= (lr * (lnsdelta_W2 + (_lambda * lnsW2)))
				lnsW1 -= (lr * (lnsdelta_W1 + (_lambda * lnsW1)))

			print('#=',split,' xlns b=','two ',' F=',xl.xlnsF,' B=',xl.xlnsB, ' batch=',batchsize, ' lr=',lr)
			lnscorrect_count = 0
			for mbatch in range(int(split / batchsize)):
				start = mbatch * batchsize
				x = x_train[start:(start + batchsize)]
				y = y_train[start:(start + batchsize)]
				if main_params['type'] == 'xlnsnp':
				  lnsx = xl.xlnsnp(np.array(xl.xlnscopy(np.array(x,dtype=np.float64))))
				if main_params['type'] == 'xlnsnpv':
				  lnsx = xl.xlnsnpv(np.array(xl.xlnscopy(np.array(x,dtype=np.float64))))
				if main_params['type'] == 'xlnsnpb':
				  lnsx = xl.xlnsnpb(np.array(xl.xlnscopy(np.array(x,dtype=np.float64))),2**2**-xl.xlnsF)
				if main_params['type'] == 'xlns':
				  lnsx = (np.array(xl.xlnscopy(np.array(x,dtype=np.float64))))
				if main_params['type'] == 'xlnsud':
				  lnsx = (np.array(xl.xlnscopy(np.array(x,dtype=np.float64),xl.xlnsud)))
				if main_params['type'] == 'xlnsv':
				  lnsx = (np.array(xl.xlnscopy(np.array(x,dtype=np.float64),xl.xlnsv)))
				if main_params['type'] == 'xlnsb':
				  lnsx = (np.array(xl.xlnscopy(np.array(x,dtype=np.float64),xl.xlnsv,2**2**-xl.xlnsF)))
				if main_params['type'] == 'float':
				  lnsx = (np.array((np.array(x,dtype=np.float64))))
				lnss1 = xl.hstack((lnsones, lnsx)) @ lnsW1
				lnsmask = (lnss1 > 0) + (leaking_coeff * (lnss1 < 0))
				lnsa1 = lnss1 * lnsmask
				lnss2 = xl.hstack((lnsones, lnsa1)) @ lnsW2
				lnscorrect_count += np.sum(np.argmax(y, axis=1) == xl.argmax(lnss2, axis=1))
			lnsaccuracy = lnscorrect_count / split
			print("train-set accuracy at epoch %d: %f" % ((1 + epoch), lnsaccuracy))
			performance['lnsacc_train'][epoch] = 100 * lnsaccuracy
			lnscorrect_count = 0  #do same # as in train set for val set
			for mbatch in range(int(split / batchsize)):
				start = mbatch * batchsize
				x = x_val[start:(start + batchsize)]
				y = y_val[start:(start + batchsize)]
				if main_params['type'] == 'xlnsnp':
				  lnsx = xl.xlnsnp(np.array(xl.xlnscopy(np.array(x,dtype=np.float64))))
				if main_params['type'] == 'xlnsnpv':
				  lnsx = xl.xlnsnpv(np.array(xl.xlnscopy(np.array(x,dtype=np.float64))))
				if main_params['type'] == 'xlnsnpb':
				  lnsx = xl.xlnsnpb(np.array(xl.xlnscopy(np.array(x,dtype=np.float64))),2**2**-xl.xlnsF)
				if main_params['type'] == 'xlns':
				  lnsx = (np.array(xl.xlnscopy(np.array(x,dtype=np.float64))))
				if main_params['type'] == 'xlnsud':
				  lnsx = (np.array(xl.xlnscopy(np.array(x,dtype=np.float64),xl.xlnsud)))
				if main_params['type'] == 'xlnsv':
				  lnsx = (np.array(xl.xlnscopy(np.array(x,dtype=np.float64),xl.xlnsv)))
				if main_params['type'] == 'xlnsb':
				  lnsx = (np.array(xl.xlnscopy(np.array(x,dtype=np.float64),xl.xlnsv,2**2**-xl.xlnsF)))
				if main_params['type'] == 'float':
				  lnsx = (np.array((np.array(x,dtype=np.float64))))
				lnss1 = xl.hstack((lnsones, lnsx)) @ lnsW1
				lnsmask = (lnss1 > 0) + (leaking_coeff * (lnss1 < 0))
				lnsa1 = lnss1 * lnsmask
				lnss2 = xl.hstack((lnsones, lnsa1)) @ lnsW2
				lnscorrect_count += np.sum(np.argmax(y, axis=1) == xl.argmax(lnss2, axis=1))
			lnsaccuracy = lnscorrect_count / split
			print("Val-set accuracy at epoch %d: %f" % ((1 + epoch), lnsaccuracy))
			performance['lnsacc_val'][epoch] = 100 * lnsaccuracy
		print("elasped time="+str(time.process_time()-start_time))
		fig = plt.figure(figsize = (16, 9)) 
		ax = fig.add_subplot(111)
		x = range(1, 1 + performance['lnsacc_train'].size)
		#ax.plot(x, performance['acc_train'], 'g')
		#ax.plot(x, performance['acc_val'], 'b')
		ax.plot(x, performance['lnsacc_train'], 'y')
		ax.plot(x, performance['lnsacc_val'], 'm')
		ax.set_xlabel('Number of Epochs')
		ax.set_ylabel('Accuracy')
		#ax.set_title('Test-set Accuracy at %.2f%%' % accuracy)
		plt.suptitle(main_params['type']+' '+str(split)+' Validation and Training MNIST Accuracies F='+str(xl.xlnsF), fontsize=14)
		ax.legend(['train', 'validation'])
		plt.grid(which='both', axis='both', linestyle='-.')

		plt.savefig('genericaccuracy.png')

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--is_training', default = True)
	parser.add_argument('--split', default = 50) #00)
	parser.add_argument('--learning_rate', default = 0.01)
	parser.add_argument('--lambda', default = 0.000)     #.001
	parser.add_argument('--minibatch_size', default = 1) #5
	parser.add_argument('--num_epoch', default = 5) #9) #12)  #40)     #10 20
	parser.add_argument('--leaking_coeff', default = 0.0078125)
	parser.add_argument('--type', default = 'float')
	args = parser.parse_args()
	main_params = vars(args)
	main(main_params)
