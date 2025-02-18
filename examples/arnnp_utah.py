#################################################
# 
# used NN code from Arnab Sanyal USC. Spring 2019
# arnnp_utah.py <- arnnp_lpvip.py <- arnabs8xlnsnp2024r.py
# uses np LPVIP LNS (use Utah's Lean-proven Taylor/cotran 
###################################################

import numpy as np
import matplotlib.pyplot as plt
#from sklearn.utils import shuffle
import argparse
import math
import os
import xlns as xl

import xlnsconf.utah_tayco_ufunc #uncomment for utah 

XlnsExactsb = True #False  #line A
XlnsExactdb = True #False  #line B
exactportion = 0  #nothing changes from above two lines exact
xl.xlnssetF(7)

#comments with xlnsnpr are points to choose redun or not

##############################
def softmax_orig(inp):
        max_vals = np.max(inp, axis=1)
        max_vals = np.reshape(max_vals, (max_vals.size, 1))
        u = np.exp(inp - max_vals)
        v = np.sum(u, axis=1)
        v.shape = v.size, 1
        u = u / v
        return u

def softmax(inp):
	max_vals = xl.xlnsnp.max(inp, axis=1)
	max_vals = xl.xlnsnp.reshape(max_vals, (xl.xlnsnp.size(max_vals), 1))
	u = xl.xlnsnp.exp(inp - max_vals)
	v = xl.xlnsnp.sum(u, axis=1)
	v = xl.xlnsnp.reshape(v, (xl.xlnsnp.size(v), 1))
	u = u / v
	return u

def comparefx(desc,f,x):
	#print(desc+" "+str(np.square(f-(x.xlns())).sum()))
	#print(input())
	return

def main(main_params):
	global XlnsExactsb,XlnsExactdb
	#XlnsExactsb = True 
	#XlnsExactdb = True
	print(" uses np Utah LNS ")

	is_training = bool(main_params['is_training'])
	leaking_coeff = float(main_params['leaking_coeff'])
	batchsize = int(main_params['minibatch_size'])
	lr = float(main_params['learning_rate'])
	num_epoch = int(main_params['num_epoch'])
	_lambda = float(main_params['lambda'])
	ones = np.array((list(np.ones((batchsize, 1)))))
	lnsones = xl.xlnsnp(np.array(xl.xlnscopy(list(np.ones((batchsize, 1))))))
		
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

		print('#=',split,' xlns b=','two ',' F=',xl.xlnsF,' B=',xl.xlnsB, ' batch=',batchsize, ' lr=',lr)
 
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
		#print('any W1 zero: '+str((W1 == 0).any()))
		#print('any W2 zero: '+str((W2 == 0).any()))
		#print('any W1~zero: '+str((abs(W1) < 1e-40).any()))
		#print('any W2~zero: '+str((abs(W2) < 1e-40).any()))
		print('min|W1|: '+str((abs(W1)).min()))
		print('min|W2|: '+str((abs(W2)).min()))

		lnsW1 = xl.xlnsnp(np.array(xl.xlnscopy(list(W1))))
		lnsW2 = xl.xlnsnp(np.array(xl.xlnscopy(list(W2))))
		#print('random weights converted to Xlns')


		delta_W1 = np.array((list(np.zeros(W1.shape))))
		delta_W2 = np.array((list(np.zeros(W2.shape))))
		lnsdelta_W1 = xl.xlnsnp(np.array(xl.xlnscopy(list(np.zeros(W1.shape)))))
		lnsdelta_W2 = xl.xlnsnp(np.array(xl.xlnscopy(list(np.zeros(W2.shape)))))
		#print('deltas initialized to Xlns zero')


		performance = {}
		performance['loss_train'] = np.zeros(num_epoch)
		performance['acc_train'] = np.zeros(num_epoch)
		performance['acc_val'] = np.zeros(num_epoch)
		performance['lnsacc_train'] = np.zeros(num_epoch)
		performance['lnsacc_val'] = np.zeros(num_epoch)

		accuracy = 0.0
		#exlnsone = xlns(1.0)

		lnsoneplrW1 = xl.xlnsnp(np.ones(W1.shape))+lr
		rlnsoneplrW1 = 1/lnsoneplrW1
		lnsoneplrW2 = xl.xlnsnp(np.ones(W2.shape))+lr
		rlnsoneplrW2 = 1/lnsoneplrW2

		for epoch in range(num_epoch):
			print('At Epoch %d:' % (1 + epoch))
			loss = 0.0
			for mbatch in range(int(split / batchsize)):
				if exactportion == 1: 
					XlnsExactsb = True 
					XlnsExactdb = True

				start = mbatch * batchsize
				x = np.array((list(x_train[start:(start + batchsize)])))
				y = np.array((list(y_train[start:(start + batchsize)])))
				lnsx = xl.xlnsnp(np.array(xl.xlnscopy(np.array(x,dtype=np.float64))))
				lnsy = xl.xlnsnp(np.array(xl.xlnscopy(np.array(y,dtype=np.float64))))
				##print(mbatch,'y=',y,lnsy)

				s1 = np.hstack((ones, x)) @ W1

				#xlnsnprA
				lnss1 = xl.xlnsnp.hstack((lnsones, lnsx)) @ lnsW1
				#lnss1 = xl.xlnsnp(xl.xlnsnpr(xl.xlnsnp.hstack((lnsones, lnsx))) @ lnsW1)

				comparefx("s1",s1,lnss1)
				#print(xl.xlnsnp.shape(lnss1))
				#print(xl.xlnsnp.shape(lnsx)) 
				#print(xl.xlnsnp.shape(lnsW1))
				##print(mbatch,s1,lnss1)
				#print('DIFF  s1=',np.sum(np.square(s1-lnss1)))
			

				mask = (s1 > 0) + (leaking_coeff * (s1 < 0))
				lnsmask = (lnss1 > 0) + (leaking_coeff * (lnss1 < 0))
				comparefx("mask",mask,lnsmask)
				#print(lnsmask[0][0:5])
				#print(lnss1[0][0:5])
				#lnsmask = (flnsS(lnss1)==1) + (leaking_coeff * (flnsS(lnss1) == -1 ))
				#print(mask,lnsmask)

				a1 = s1 * mask
				lnsa1 = lnss1 * lnsmask
				comparefx("a1",a1,lnsa1)
				##print(mbatch,a1,lnsa1)
				#print('diff  a1=',np.sum(np.square(a1-lnsa1)))

				if exactportion == 2: 
					XlnsExactsb = True 
					XlnsExactdb = True

				s2 = np.hstack((ones, a1)) @ W2

				#xlnsnprB
				lnss2 =                     xl.xlnsnp.hstack((lnsones, lnsa1)) @ lnsW2
				#lnss2 = xl.xlnsnp(xl.xlnsnpr(xl.xlnsnp.hstack((lnsones, lnsa1))) @ lnsW2)

				comparefx("s2",s2,lnss2)
				#print('diff  s2=',np.sum(np.square(s2-lnss2)))

				if exactportion == 3: 
					XlnsExactsb = True 
					XlnsExactdb = True

				a2 = softmax_orig(s2)
				lnsa2 = softmax(lnss2)
				comparefx("a2",a2,lnsa2)
				##print(mbatch,'a2=',a2,lnsa2)
				#print('diff  a2=',np.sum(np.square(a2-lnsa2)))

				cat_cross_ent = np.log(a2) * y
				#cat_cross_ent[np.isnan(cat_cross_ent)] = 0
				loss -= np.sum(cat_cross_ent)

				#print(str(a2)+str(y))
				grad_s2 = (a2 - y) / batchsize
				#print(type(grad_s2))
				#print(grad_s2)
				#print(str(lnsa2)+str(lnsy))
				lnsgrad_s2 = (lnsa2 - lnsy) / batchsize
				#print(type(lnsgrad_s2))
				#print(lnsgrad_s2)
				#print(grad_s2-(lnsgrad_s2.xlns()))
				comparefx("grad_s2",grad_s2,lnsgrad_s2)
				##print(mbatch,'gs2=',grad_s2,lnsgrad_s2)
				#print('diff gs2=',np.sum(np.square(grad_s2-lnsgrad_s2)))

				if exactportion == 3: 
					XlnsExactsb = False 
					XlnsExactdb = False
				
				grad_a1 = grad_s2 @ W2[1:].T

				#xlnsnprC
				lnsgrad_a1 =                     lnsgrad_s2 @ xl.xlnsnp.transpose(lnsW2[1:])
				#lnsgrad_a1 = xl.xlnsnp(xl.xlnsnpr(lnsgrad_s2) @ xl.xlnsnp.transpose(lnsW2[1:]))
				comparefx("grad_a1",grad_a1,lnsgrad_a1)
				##print(mbatch,'ga1=',grad_a1,lnsgrad_a1)
				#print('diff ga1=',np.sum(np.square(grad_a1-lnsgrad_a1)))
				
				delta_W2 = np.hstack((ones, a1)).T @ grad_s2
				#print(xl.xlnsnp.hstack((lnsones, lnsa1)))
				#print(lnsgrad_s2)
				#print(xl.xlnsnp.transpose(xl.xlnsnp.hstack((lnsones, lnsa1)))) 
				lnsdelta_W2 = xl.xlnsnp.transpose(xl.xlnsnp.hstack((lnsones, lnsa1))) @ lnsgrad_s2

				if exactportion == 2: 
					XlnsExactsb = False 
					XlnsExactdb = False


				grad_s1 = mask * grad_a1
				lnsgrad_s1 = lnsmask * lnsgrad_a1
				comparefx("grad_s1",grad_s1,lnsgrad_s1)
				#print(str(type(lnsgrad_s1))+str(type(lnsmask))+str(type(lnsgrad_a1)))
				#print('diff gs1=',np.sum(np.square(grad_s1-lnsgrad_s1)))

				delta_W1 = np.hstack((ones, x)).T @ grad_s1
				lnsdelta_W1 = xl.xlnsnp.transpose(xl.xlnsnp.hstack((lnsones, lnsx))) @ lnsgrad_s1

				#W2 -= (lr * (delta_W2 + (_lambda * W2)))
				#W1 -= (lr * (delta_W1 + (_lambda * W1)))
				#lnsW2 -= (lr * (lnsdelta_W2 + (_lambda * lnsW2)))
				#lnsW1 -= (lr * (lnsdelta_W1 + (_lambda * lnsW1)))

				W2 -= (lr * delta_W2) # lambda=0
				W1 -= (lr * delta_W1)
				lnsW2 -= (lr * lnsdelta_W2)
				lnsW1 -= (lr * lnsdelta_W1)

				if mbatch%10==0:
				        print(mbatch)
				        #print('any W1 zero: '+str((W1 == 0).any()))
				        #print('any W2 zero: '+str((W2 == 0).any()))
				        #print('any W1~zero: '+str((abs(W1) < 1e-40).any()))
				        #print('any W2~zero: '+str((abs(W2) < 1e-40).any()))

				        if mbatch%100==0:
				            print('lnsx #0s:   '+str(((lnsx.xlns())==0).sum()))
				            print('lnsa1 #0s:  '+str(((lnsa1.xlns())==0).sum()))
				            print('lnsgs2 #0s: '+str(((lnsgrad_s2.xlns())==0).sum()))
				            print('min|W1|:    '+str((abs(W1)).min()))
				            print('min|lnsW1|: '+str((abs(lnsW1.xlns())).min()))

				            print('min|W2|:    '+str((abs(W2)).min()))
				            print('min|lnsW2|: '+str((abs(lnsW2.xlns())).min()))

				            print('max|W1|:    '+str((abs(W1)).max()))
				            print('max|lnsW1|: '+str((abs(lnsW1.xlns())).max()))

				            print('max|W2|:    '+str((abs(W2)).max()))
				            print('max|lnsW2|: '+str((abs(lnsW2.xlns())).max()))

				            #print('+0s delta_W1:    '+str((np.sign(delta_W1)==0).sum()))
				            #print('+1s delta_W1:    '+str((np.sign(delta_W1)==1).sum()))
				            #print('+1s lnsdelta_W1: '+str((xl.xlnsnp.sign(lnsdelta_W1).nd==0).sum()))
				            #print('-1s delta_W1:    '+str((np.sign(delta_W1)==-1).sum()))
				            #print('-1s lnsdelta_W1: '+str((xl.xlnsnp.sign(lnsdelta_W1).nd==1).sum()))
				            #print(np.size(delta_W1))
				            #print('+0s delta_W2:    '+str((np.sign(delta_W2)==0).sum()))
				            #print('+1s delta_W2:    '+str((np.sign(delta_W2)==1).sum()))
				            #print('+1s lnsdelta_W2: '+str((xl.xlnsnp.sign(lnsdelta_W2).nd==0).sum()))
				            #print('-1s delta_W2:    '+str((np.sign(delta_W2)==-1).sum()))
				            #print('-1s lnsdelta_W2: '+str((xl.xlnsnp.sign(lnsdelta_W2).nd==1).sum()))
				            #print(np.size(delta_W2))


				#print(mbatch,' diff W1,W2=',np.sum(np.square(W1-lnsW1.xlns())),np.sum(np.square(W2-lnsW2.xlns())))
				if exactportion == 1: 
					XlnsExactsb = False 
					XlnsExactdb = False
			loss /= split
			performance['loss_train'][epoch] = loss
			print('Loss at epoch %d: %f' %((1 + epoch), loss))
			#print('type W1 W2 s1 a1 s2 a2:',type(W1[0][0]),type(W2[0][0]),type(s1[0]),type(a1[0]),type(s2[0]),type(a2[0]),type(x[0]))
			#print('diff W1,W2=',np.sum(np.square(W1-lnsW1.xlns())),np.sum(np.square(W2-lnsW2.xlns())))
			#exit()

			print('#=',split,' xlns b=','two ',' F=',xl.xlnsF,' B=',xl.xlnsB, ' batch=',batchsize, ' lr=',lr)
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

				lnsx = xl.xlnsnp(np.array(xl.xlnscopy(np.array(x,dtype=np.float64))))
				lnss1 = xl.xlnsnp.hstack((lnsones, lnsx)) @ lnsW1
				lnsmask = (lnss1 > 0) + (leaking_coeff * (lnss1 < 0))
				#lnsmask = (lnss1 > 0) + (leaking_coeff * (lnss1 < 0))
				#lnsmask = (flnsS(lnss1)==1) + (leaking_coeff * (flnsS(lnss1) == -1 ))
				lnsa1 = lnss1 * lnsmask
				lnss2 = xl.xlnsnp.hstack((lnsones, lnsa1)) @ lnsW2

				#print('DIFF  s1=',np.sum(np.square(s1-lnss1)))
				#print('diff  a1=',np.sum(np.square(a1-lnsa1)))
				#print('diff  s2=',np.sum(np.square(s2-lnss2)))

				lnscorrect_count += np.sum(np.argmax(y, axis=1) == xl.xlnsnp.argmax(lnss2, axis=1))
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
			for mbatch in range(int(split / batchsize)):

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

				lnsx = xl.xlnsnp(np.array(xl.xlnscopy(np.array(x,dtype=np.float64))))
				lnss1 = xl.xlnsnp.hstack((lnsones, lnsx)) @ lnsW1
				lnsmask = (lnss1 > 0) + (leaking_coeff * (lnss1 < 0))
				#lnsmask = (flnsS(lnss1)==1) + (leaking_coeff * (flnsS(lnss1) == -1 ))
				lnsa1 = lnss1 * lnsmask
				lnss2 = xl.xlnsnp.hstack((lnsones, lnsa1)) @ lnsW2

				#print('DIFF  s1=',np.sum(np.square(s1-lnss1)))
				#print('diff  a1=',np.sum(np.square(a1-lnsa1)))
				#print('diff  s2=',np.sum(np.square(s2-lnss2)))

				lnscorrect_count += np.sum(np.argmax(y, axis=1) == xl.xlnsnp.argmax(lnss2, axis=1))
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
		plt.suptitle('FP versus LNS xlnsnp_lpvip '+str(split)+' Validation and Training MNIST Accuracies F='+str(xl.xlnsF), fontsize=14)
		ax.legend(['FP train', 'FP validation','LNS train', 'LNS validation'])
		plt.grid(which='both', axis='both', linestyle='-.')

		plt.savefig('plotmnist_xlnsnp_lpvip.png')

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--is_training', default = True)
	parser.add_argument('--split', default = 50) #00)
	parser.add_argument('--learning_rate', default = 0.01)
	parser.add_argument('--lambda', default = 0.000)     #.001
	parser.add_argument('--minibatch_size', default = 1) #5
	parser.add_argument('--num_epoch', default = 5) #9) #12)  #40)     #10 20
	parser.add_argument('--leaking_coeff', default = 0.0078125)
	args = parser.parse_args()
	main_params = vars(args)
	main(main_params)
