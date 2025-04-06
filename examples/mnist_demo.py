# Based on arn_generic.py which uses NN code from Arnab Sanyal USC. Spring 2019
# This fixes bugs in arn_generic and cleans up the code

import os
import time
import argparse

from matplotlib import pyplot as plt
import numpy as np
import xlns as xl

# Functions to convert from a numpy array to a given xlns object
converters = {
	'xlnsnp': lambda x: xl.xlnsnp(x),
	'xlnsnpv': lambda x, f=None: xl.xlnsnpv(x, setF=f),
	'xlnsnpb': lambda x, b: xl.xlnsnpb(x, setB=b),
	'xlns': lambda x: np.array(xl.xlnscopy(x)),
	'xlnsud': lambda x: np.array(xl.xlnscopy(x, xl.xlnsud)),
	'xlnsv': lambda x, f=None: np.array(xl.xlnscopy(x, xl.xlnsv, setFB=f)),
	'xlnsb': lambda x, b: np.array(xl.xlnscopy(x, xl.xlnsb, setFB=b)),
	'float': lambda x: x
}

# Splits arrays into batches (returns the length of the batch)
# so we can handle a final smaller batch if batchsize doesn't
# divide the length of the arrays.
def batch_generator(x, y, batch_size):
	for start in range(0, len(x), batch_size):
		end = start + batch_size
		batch_x, batch_y = x[start:end], y[start:end]
		yield batch_x, batch_y, len(batch_x)

def softmax(inp):
	max_vals = inp.max(axis=1)
	max_vals = xl.reshape(max_vals,(xl.size(max_vals), 1))
	u = xl.exp(inp - max_vals)
	v = u.sum(axis=1)
	v = v.reshape((xl.size(v), 1))
	u = u / v
	return u

def main(main_params):

	is_training = main_params['is_training']
	leaking_coeff = main_params['leaking_coeff']
	batchsize = main_params['minibatch_size']
	lr = main_params['learning_rate']
	num_epoch = main_params['num_epoch']
	_lambda = main_params['lambda']
	data_type = main_params['type']
	precision = main_params['precision']
	weights_precision = main_params['weights_precision']

	# Sets the default precision used for fixed precision xlns objects
	xl.xlnssetF(precision)

	mnist_path = main_params['mnist']
	if not os.path.isfile(mnist_path):
		raise FileNotFoundError(f"Couldn't find '{mnist_path}'. Ensure it's in the right directory.")
	with np.load(mnist_path) as data:
		x_train, y_train = data['train_data'], data['train_labels']
		x_test, y_test = data['test_data'], data['test_labels']

	weightin_path = main_params['weight_in']
	weight_scale = main_params['weight_scale']
	if os.path.isfile(weightin_path):
		print(f"Using existing weights from {weightin_path}")
		with np.load("./weightin.npz") as data:
			W1 = data["W1"]
			W2 = data["W2"]
	else:
		print(f"Weight input file not found, generating random weights (scale: {weight_scale})")
		W1 = np.random.normal(0, weight_scale, (785, 100))
		W2 = np.random.normal(0, weight_scale, (101, 10))
		weightout_path = main_params['weight_out']
		np.savez_compressed(weightout_path, W1=W1, W2=W2)

	# Hack to get rid of large if statements. Some of the converters require us to 
	# specify a precision (xlnsv, xlnsnpv) or a base (xlnsb, xlnsnpb) so we must
	# pass additional arguments to them. 
	if data_type in ['xlnsnp', 'xlns', 'xlnsud', 'float']:
		weights_fb, input_fb = [], []
	elif data_type in ['xlnsnpv', 'xlnsv']:
		weights_fb, input_fb = [weights_precision], []
	elif data_type in ['xlnsnpb', 'xlnsb']:
		weights_fb, input_fb = [2.0**(2**(-weights_precision))], [2.0**(2**(-xl.xlnsF))]
	converter = converters[data_type]
	lnsW1 = converter(W1, *weights_fb)
	lnsW2 = converter(W2, *weights_fb)
	lnsones_full = converter(np.ones((batchsize, 1)), *input_fb)

	val_split = main_params['val_split']
	test_len = len(x_test)
	if val_split > test_len:
		print(f"val-split > num of validation samples, using all samples for training ({test_len})")
		val_split = test_len
	x_val, y_val = x_test[:val_split], y_test[:val_split]

	if is_training:

		split = main_params['split']
		train_len = len(x_train)
		if split > train_len:
			print(f"split > num of training samples, using all samples for training ({train_len})")
			split = train_len
		x_train, y_train = x_train[:split], y_train[:split]

		print(f"Traing size = {split} | xlns b = 2 | F = {xl.xlnsF} | B = {xl.xlnsB} | batch = {batchsize} | lr = {lr}")
		performance = {
			'lnsacc_train': np.zeros(num_epoch),
			'lnsacc_val': np.zeros(num_epoch)
		}
		start_time = time.perf_counter()
		for epoch in range(num_epoch):

			# Train on the training set with gradient descent
			print(f"At Epoch {epoch + 1}:")
			for x, y, curr_batchsize in batch_generator(x_train, y_train, batchsize):

				lnsx, lnsy = converter(x, *input_fb), converter(y, *input_fb)
				lnsones = lnsones_full if curr_batchsize == batchsize else converter(np.ones((curr_batchsize, 1)), *input_fb)

				lnss1 = xl.hstack((lnsones, lnsx)) @ lnsW1
				lnsmask = (lnss1 > 0) + (leaking_coeff * (lnss1 < 0))
				lnsa1 = lnss1 * lnsmask
				lnss2 = xl.hstack((lnsones, lnsa1)) @ lnsW2
				lnsa2 = softmax(lnss2)

				lnsgrad_s2 = (lnsa2 - lnsy) / curr_batchsize
				lnsgrad_a1 = lnsgrad_s2 @ xl.transpose(lnsW2[1:])
				lnsdelta_W2 = xl.transpose(xl.hstack((lnsones, lnsa1))) @ lnsgrad_s2
				lnsgrad_s1 = lnsmask * lnsgrad_a1
				lnsdelta_W1 = xl.transpose(xl.hstack((lnsones, lnsx))) @ lnsgrad_s1
				lnsW2 -= (lr * (lnsdelta_W2 + (_lambda * lnsW2)))
				lnsW1 -= (lr * (lnsdelta_W1 + (_lambda * lnsW1)))

			# Test accuracy on the training set
			lnscorrect_count = 0
			for x, y, curr_batchsize in batch_generator(x_train, y_train, batchsize):

				lnsx, lnsy = converter(x, *input_fb), converter(y, *input_fb)
				lnsones = lnsones_full if curr_batchsize == batchsize else converter(np.ones((curr_batchsize, 1)), *input_fb)

				lnss1 = xl.hstack((lnsones, lnsx)) @ lnsW1
				lnsmask = (lnss1 > 0) + (leaking_coeff * (lnss1 < 0))
				lnsa1 = lnss1 * lnsmask
				lnss2 = xl.hstack((lnsones, lnsa1)) @ lnsW2
				lnscorrect_count += np.sum(np.argmax(y, axis=1) == xl.argmax(lnss2, axis=1))

			lnsaccuracy = lnscorrect_count / split
			performance['lnsacc_train'][epoch] = 100 * lnsaccuracy
			print(f"Training set accuracy after epoch {epoch + 1}: {lnsaccuracy}")

			# Test accuracy on the validation set
			lnscorrect_count = 0
			for x, y, curr_batchsize in batch_generator(x_val, y_val, batchsize):

				lnsx = converter(x, *input_fb)
				lnsones = lnsones_full if curr_batchsize == batchsize else converter(np.ones((curr_batchsize, 1)), *input_fb)

				lnss1 = xl.hstack((lnsones, lnsx)) @ lnsW1
				lnsmask = (lnss1 > 0) + (leaking_coeff * (lnss1 < 0))
				lnsa1 = lnss1 * lnsmask
				lnss2 = xl.hstack((lnsones, lnsa1)) @ lnsW2
				lnscorrect_count += np.sum(np.argmax(y, axis=1) == xl.argmax(lnss2, axis=1))

			lnsaccuracy = lnscorrect_count / val_split
			performance['lnsacc_val'][epoch] = 100 * lnsaccuracy
			print(f"Validation set accuracy after epoch {epoch + 1}: {lnsaccuracy}")

		time_elapsed = time.perf_counter() - start_time
		print(f"Elapsed time: {time_elapsed}")

		plot_figure = main_params['figure']
		figure_path = main_params['figure_path']
		if plot_figure:

			fig = plt.figure(figsize = (16, 9)) 
			ax = fig.add_subplot(111)
			x = range(1, performance['lnsacc_train'].size + 1)

			ax.plot(x, performance['lnsacc_train'], 'y')
			ax.plot(x, performance['lnsacc_val'], 'm')
			ax.set_xlabel('Number of Epochs')
			ax.set_ylabel('Accuracy')

			plt.suptitle(f"'{data_type}' Validation and Training MNIST Accuracies", fontsize=14)
			ax.legend(['train', 'validation'])
			plt.grid(which='both', axis='both', linestyle='-.')
			plt.savefig(figure_path)

	else:

		# Test accuracy on the validation set without training
		lnscorrect_count = 0
		for x, y, curr_batchsize in batch_generator(x_val, y_val, batchsize):

			lnsx = converter(x, *input_fb)
			lnsones = lnsones_full if curr_batchsize == batchsize else converter(np.ones((curr_batchsize, 1)), *input_fb)

			lnss1 = xl.hstack((lnsones, lnsx)) @ lnsW1
			lnsmask = (lnss1 > 0) + (leaking_coeff * (lnss1 < 0))
			lnsa1 = lnss1 * lnsmask
			lnss2 = xl.hstack((lnsones, lnsa1)) @ lnsW2
			lnscorrect_count += np.sum(np.argmax(y, axis=1) == xl.argmax(lnss2, axis=1))

		lnsaccuracy = lnscorrect_count / val_split
		print(f"Validation set accuracy: {lnsaccuracy}")

if __name__ == "__main__":

	defaults = {
		'is_training': True,
		'split': 50,
		'val_split': 50,
		'mnist': './mnist.npz',
		'weight_in': './weightin.npz',
		'weight_out': './weightout.npz',
		'weight_scale': 0.1,
		'figure': True,
		'figure_path': 'genericaccuracy.png',
		'learning_rate': 0.01,
		'lambda_param': 0.000,
		'minibatch_size': 1,
		'num_epoch': 5,
		'leaking_coeff': 0.0078125,
		'type': 'float',
		'precision': 10,
		'weights_precision': 6
	}

	parser = argparse.ArgumentParser(description="Training configuration")
	parser.add_argument('--is_training', type=bool, default=defaults['is_training'],
						help=f"Whether to train or test the model (default: {defaults['is_training']})")
	parser.add_argument('--split', type=int, default=defaults['split'],
						help=f"Split value (default: {defaults['split']})")
	parser.add_argument('--val_split', type=int, default=defaults['val_split'],
						help=f"Validation split value (default: {defaults['val_split']})")
	parser.add_argument('--mnist', type=str, default=defaults['mnist'],
					 	help=f"Path to the MNIST data file (default: '{defaults['mnist']}')")
	parser.add_argument('--weight_in', type=str, default=defaults['weight_in'],
                        help=f"Path to the .npz file containing input weights (default: '{defaults['weight_in']}')")
	parser.add_argument('--weight_out', type=str, default=defaults['weight_out'],
                        help=f"Path where newly generated weights will be saved (default: '{defaults['weight_out']}')")
	parser.add_argument('--weight_scale', type=float, default=defaults['weight_scale'],
                        help=f"Standard deviation for initialized weights (default: {defaults['weight_scale']})")
	parser.add_argument('--figure', type=bool, default=defaults['figure'],
					 	help=f"Whether to generate a figure of training convergence (default: {defaults['figure']})")
	parser.add_argument('--figure_path', type=str, default=defaults['figure_path'],
					 	help=f"Path where the figure will be saved to (default: {defaults['figure_path']})")
	parser.add_argument('--learning_rate', type=float, default=defaults['learning_rate'],
						help=f"Learning rate (default: {defaults['learning_rate']})")
	parser.add_argument('--lambda', type=float, default=defaults['lambda_param'],
						help=f"Regularization parameter lambda (default: {defaults['lambda_param']})")
	parser.add_argument('--minibatch_size', type=int, default=defaults['minibatch_size'],
						help=f"Minibatch size (default: {defaults['minibatch_size']})")
	parser.add_argument('--num_epoch', type=int, default=defaults['num_epoch'],
						help=f"Number of epochs (default: {defaults['num_epoch']})")
	parser.add_argument('--leaking_coeff', type=float, default=defaults['leaking_coeff'],
						help=f"Leaking coefficient (default: {defaults['leaking_coeff']})")
	parser.add_argument('--type', type=str, default=defaults['type'],
						help=f"Data type parameter (default: '{defaults['type']}')")
	parser.add_argument('--precision', type=int, default=defaults['precision'],
					 	help=f"Precision for feedforward values (default: '{defaults['precision']}')")
	parser.add_argument('--weights_precision', type=int, default=defaults['weights_precision'],
					 	help=f"Precision for weight values (default: '{defaults['weights_precision']}')")

	args = parser.parse_args()
	main_params = vars(args)
	main(main_params)