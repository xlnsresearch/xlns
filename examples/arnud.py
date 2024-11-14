###################################################
# 
# used NN code from Arnab Sanyal USC. Spring 2019
# arnud.py <- arnabs8xlnsudAiNvGd2024.py
# slow scalar "user def" LNS with option for lpvip or fracnorm
###################################################

import numpy as np
import matplotlib.pyplot as plt
import argparse
import math
import os
import xlns as xl

#import xlnsconf.lpvip_ufunc       #uncomment for lpvip
import xlnsconf.xlnsudFracnorm  #uncomment for Zhang/Han or Paliouras fractional normalize
xlnsconf.xlnsudFracnorm.ilog2 = xlnsconf.xlnsudFracnorm.ipallog2 #also uncomment only for Paliouras
xlnsconf.xlnsudFracnorm.ipow2 = xlnsconf.xlnsudFracnorm.ipalpow2

xl.xlnssetF(7)

def softmax(inp):

	# inp_shape = inp.shape
	max_vals = np.max(inp, axis=1)
	max_vals.shape = max_vals.size, 1
	u = np.exp(inp - max_vals)
	v = np.sum(u, axis=1)
	v.shape = v.size, 1
	u = u / v
	return u

def main(main_params):

	print(' slow scalar "user def" LNS with option for lpvip or fracnorm')
	is_training = bool(main_params['is_training'])
	leaking_coeff = float(main_params['leaking_coeff'])
	batchsize = int(main_params['minibatch_size'])
	lr = float(main_params['learning_rate'])
	num_epoch = int(main_params['num_epoch'])
	_lambda = float(main_params['lambda'])
	ones = np.array((list(np.ones((batchsize, 1)))))
	lnsones = np.array(xl.xlnscopy(list(np.ones((batchsize, 1))),xl.xlnsud))
		
	if is_training:
		# load mnist data and split into train and test sets
		# one-hot encoded target column

		file = np.load('./mnist.npz', 'r') # dataset
		x_train = ((file['train_data']))
		y_train = np.array((file['train_labels']))
		print('type xtrain ytrain:',type(x_train[0]),type(y_train[0]))
		x_test = np.array((file['test_data']))
		y_test = np.array((file['test_labels']))
		#x_train, y_train = shuffle(x_train, y_train)
		#x_test, y_test = shuffle(x_test, y_test)
		file.close()


		split = int(main_params['split'])
		x_val = x_train[split:]
		print('xval ',x_val.shape[0])

		print('#=',split,' F=',xl.xlnsF,' B=',xl.xlnsB, ' batch=',batchsize, ' lr=',lr)

		y_val = y_train[split:]
		y_train = y_train[:split]
		x_train = x_train[:split]

		# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

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
		lnsW1 = np.array(xl.xlnscopy(list(W1),xl.xlnsud))
		lnsW2 = np.array(xl.xlnscopy(list(W2),xl.xlnsud))
		#print('random weights converted to Xlns')


		delta_W1 = np.array((list(np.zeros(W1.shape))))
		delta_W2 = np.array((list(np.zeros(W2.shape))))
		lnsdelta_W1 = np.array(xl.xlnscopy(list(np.zeros(W1.shape)),xl.xlnsud))
		lnsdelta_W2 = np.array(xl.xlnscopy(list(np.zeros(W2.shape)),xl.xlnsud))
		#print('deltas initialized to Xlns zero')


		performance = {}
		performance['loss_train'] = np.zeros(num_epoch)
		performance['acc_train'] = np.zeros(num_epoch)
		performance['acc_val'] = np.zeros(num_epoch)
		performance['lnsacc_train'] = np.zeros(num_epoch)
		performance['lnsacc_val'] = np.zeros(num_epoch)

		accuracy = 0.0
		flnsone = xl.xlnsud(1.0)

		for epoch in range(num_epoch):
			print('At Epoch %d:' % (1 + epoch))
			loss = 0.0
			for mbatch in range(int(split / batchsize)):
				start = mbatch * batchsize
				x = np.array((list(x_train[start:(start + batchsize)])))
				y = np.array((list(y_train[start:(start + batchsize)])))
				lnsx = np.array(xl.xlnscopy(np.array(x,dtype=np.float64),xl.xlnsud))
				lnsy = y
				##print(mbatch,'y=',y,lnsy)

				s1 = np.hstack((ones, x)) @ W1
				lnss1 = np.hstack((lnsones, lnsx)) @ lnsW1
				##print(mbatch,s1,lnss1)
				#print('DIFF  s1=',np.sum(np.square(s1-lnss1)))
			

				mask = (s1 > 0) + (leaking_coeff * (s1 < 0))
				lnsmask = (lnss1 > 0) + (leaking_coeff * (lnss1 < 0))
				#lnsmask = (flnsS(lnss1)==1) + (leaking_coeff * (flnsS(lnss1) == -1 ))
				#print(mask,lnsmask)

				a1 = s1 * mask
				lnsa1 = lnss1 * lnsmask
				##print(mbatch,a1,lnsa1)
				#print('diff  a1=',np.sum(np.square(a1-lnsa1)))

				s2 = np.hstack((ones, a1)) @ W2
				lnss2 = np.hstack((lnsones, lnsa1)) @ lnsW2
				#print('diff  s2=',np.sum(np.square(s2-lnss2)))

				a2 = softmax(s2)
				lnsa2 = softmax(lnss2)
				##print(mbatch,'a2=',a2,lnsa2)
				#print('diff  a2=',np.sum(np.square(a2-lnsa2)))

				cat_cross_ent = np.log(a2) * y
				#cat_cross_ent[np.isnan(cat_cross_ent)] = 0
				loss -= np.sum(cat_cross_ent)

				grad_s2 = (a2 - y) / batchsize
				lnsgrad_s2 = (lnsa2 - lnsy) / batchsize
				##print(mbatch,'gs2=',grad_s2,lnsgrad_s2)
				#print('diff gs2=',np.sum(np.square(grad_s2-lnsgrad_s2)))
				
				grad_a1 = grad_s2 @ W2[1:].T
				lnsgrad_a1 = lnsgrad_s2 @ lnsW2[1:].T
				##print(mbatch,'ga1=',grad_a1,lnsgrad_a1)
				#print('diff ga1=',np.sum(np.square(grad_a1-lnsgrad_a1)))
				
				delta_W2 = np.hstack((ones, a1)).T @ grad_s2
				lnsdelta_W2 = np.hstack((lnsones, lnsa1)).T @ lnsgrad_s2

				grad_s1 = mask * grad_a1
				lnsgrad_s1 = lnsmask * lnsgrad_a1
				#print('diff gs1=',np.sum(np.square(grad_s1-lnsgrad_s1)))

				delta_W1 = np.hstack((ones, x)).T @ grad_s1
				lnsdelta_W1 = np.hstack((ones, x)).T @ lnsgrad_s1

				#W2 -= (lr * (delta_W2 + (_lambda * W2)))
				#W1 -= (lr * (delta_W1 + (_lambda * W1)))
				#lnsW2 -= (lr * (lnsdelta_W2 + (_lambda * lnsW2)))
				#lnsW1 -= (lr * (lnsdelta_W1 + (_lambda * lnsW1)))
				W2 -= (lr * delta_W2) # lambda=0
				W1 -= (lr * delta_W1)
				lnsW2 -= (lr * lnsdelta_W2)
				lnsW1 -= (lr * lnsdelta_W1)

				print(mbatch,' diff W1,W2=',np.sum(np.square(W1-lnsW1)),np.sum(np.square(W2-lnsW2)))
			loss /= split
			performance['loss_train'][epoch] = loss
			print('Loss at epoch %d: %f' %((1 + epoch), loss))
			#print('type W1 W2 s1 a1 s2 a2:',type(W1[0][0]),type(W2[0][0]),type(s1[0]),type(a1[0]),type(s2[0]),type(a2[0]),type(x[0]))
			print('diff W1,W2=',np.sum(np.square(W1-lnsW1)),np.sum(np.square(W2-lnsW2)))
			#exit()

			print('#=',split,' F=',xl.xlnsF,' B=',xl.xlnsB, ' batch=',batchsize, ' lr=',lr)
			correct_count = 0
			lnscorrect_count = 0
			for mbatch in range(int(split / batchsize)):

				start = mbatch * batchsize
				x = x_train[start:(start + batchsize)]
				y = y_train[start:(start + batchsize)]

				s1 = np.hstack((ones, x)) @ W1
				###################################################
				mask = (s1 > 0) + (leaking_coeff * (s1 < 0))
				###################################################
				a1 = s1 * mask
				s2 = np.hstack((ones, a1)) @ W2

				correct_count += np.sum(np.argmax(y, axis=1) == np.argmax(s2, axis=1))

				lnsx = np.array(xl.xlnscopy(np.array(x,dtype=np.float64),xl.xlnsud))
				lnss1 = np.hstack((lnsones, lnsx)) @ lnsW1
				lnsmask = (lnss1 > 0) + (leaking_coeff * (lnss1 < 0))
				#lnsmask = (flnsS(lnss1)==1) + (leaking_coeff * (flnsS(lnss1) == -1 ))
				lnsa1 = lnss1 * lnsmask
				lnss2 = np.hstack((ones, lnsa1)) @ lnsW2

				#print('DIFF  s1=',np.sum(np.square(s1-lnss1)))
				#print('diff  a1=',np.sum(np.square(a1-lnsa1)))
				#print('diff  s2=',np.sum(np.square(s2-lnss2)))

				lnscorrect_count += np.sum(np.argmax(y, axis=1) == np.argmax(lnss2, axis=1))
				#lnscorrect_count += np.sum(np.argmax(y, axis=1) == np.argmax(np.add(2*np.max(np.absolute(lnss2)),lnss2)))
				#if (np.argmax(s2, axis=1)!=np.argmax(np.add(2*np.max(np.absolute(lnss2)),lnss2))):
				#     print(s2,lnss2)
				#     print(mbatch,'ne ',np.argmax(s2, axis=1),np.argmax(np.add(2*np.max(np.absolute(lnss2)),lnss2)))

			accuracy = correct_count / split
			lnsaccuracy = lnscorrect_count / split
			print("Train-set accuracy at epoch %d: %f" % ((1 + epoch), accuracy))
			print("LNS train-set accuracy at epoch %d: %f" % ((1 + epoch), lnsaccuracy))
			performance['acc_train'][epoch] = 100 * accuracy
			performance['lnsacc_train'][epoch] = 100 * lnsaccuracy

			correct_count = 0
			lnscorrect_count = 0  #do same # as in train set for val set
			if False:
			#for mbatch in range(int(split / batchsize)):

				start = mbatch * batchsize
				x = x_val[start:(start + batchsize)]
				y = y_val[start:(start + batchsize)]

				s1 = np.hstack((ones, x)) @ W1
				###################################################
				mask = (s1 > 0) + (leaking_coeff * (s1 < 0))
				###################################################
				a1 = s1 * mask
				s2 = np.hstack((ones, a1)) @ W2

				correct_count += np.sum(np.argmax(y, axis=1) == np.argmax(s2, axis=1))

				lnsx = np.array(xl.xlnscopy(np.array(x,dtype=np.float64),xl.xlnsud))
				lnss1 = np.hstack((lnsones, lnsx)) @ lnsW1
				lnsmask = (lnss1 > 0) + (leaking_coeff * (lnss1 < 0))
				#lnsmask = (flnsS(lnss1)==1) + (leaking_coeff * (flnsS(lnss1) == -1 ))
				lnsa1 = lnss1 * lnsmask
				lnss2 = np.hstack((ones, lnsa1)) @ lnsW2

				#print('DIFF  s1=',np.sum(np.square(s1-lnss1)))
				#print('diff  a1=',np.sum(np.square(a1-lnsa1)))
				#print('diff  s2=',np.sum(np.square(s2-lnss2)))

				lnscorrect_count += np.sum(np.argmax(y, axis=1) == np.argmax(lnss2, axis=1))
				#lnscorrect_count += np.sum(np.argmax(y, axis=1) == np.argmax(np.add(2*np.max(np.absolute(lnss2)),lnss2)))
				#if (np.argmax(s2, axis=1)!=np.argmax(np.add(2*np.max(np.absolute(lnss2)),lnss2))):
				#     print(s2,lnss2)
				#     print(mbatch,'ne ',np.argmax(s2, axis=1),np.argmax(np.add(2*np.max(np.absolute(lnss2)),lnss2)))

			accuracy = correct_count / split
			lnsaccuracy = lnscorrect_count / split
			print("Val-set accuracy at epoch %d: %f" % ((1 + epoch), accuracy))
			print("LNS Val-set accuracy at epoch %d: %f" % ((1 + epoch), lnsaccuracy))
			performance['acc_val'][epoch] = 100 * accuracy
			performance['lnsacc_val'][epoch] = 100 * lnsaccuracy

			#return
			#exit()

		#np.savez_compressed('./lin_model_MNIST.npz', W1=W1, W2=W2, loss_train=performance['loss_train'], \
		#	acc_train=performance['acc_train'], acc_val=performance['acc_val'])
		fig = plt.figure(figsize = (16, 9)) 
		ax = fig.add_subplot(111)
		x = range(1, 1 + performance['loss_train'].size)
		ax.plot(x, performance['acc_train'], 'g')
		ax.plot(x, performance['acc_val'], 'b')
		ax.plot(x, performance['lnsacc_train'], 'y')
		ax.plot(x, performance['lnsacc_val'], 'm')
		ax.set_xlabel('Number of Epochs')
		ax.set_ylabel('Accuracy')
		#ax.set_title('Test-set Accuracy at %.2f%%' % accuracy)
		plt.suptitle('FP versus LNS xlnsud Validation and Training MNIST Accuracies', fontsize=14)
		ax.legend(['FP train', 'FP validation','LNS train', 'LNS validation'])
		plt.grid(which='both', axis='both', linestyle='-.')

		plt.savefig('plotmnist_ud.png')

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--is_training', default = True)
	parser.add_argument('--split', default = 50)
	parser.add_argument('--learning_rate', default = 0.01)
	parser.add_argument('--lambda', default = 0.000)     #.001
	parser.add_argument('--minibatch_size', default = 1) #5
	parser.add_argument('--num_epoch', default = 3)     #10 20
	parser.add_argument('--leaking_coeff', default = 0.0078125)
	args = parser.parse_args()
	main_params = vars(args)
	main(main_params)
