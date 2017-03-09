
import numpy as np 
import theano
from theano import tensor as T 
from theano import config
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
from theano.tensor.nnet.bn import batch_normalization as BN

from resnet_funcs import *





lr = np.asarray(0.0001,dtype='float32')
n_epochs = 10




def build_model(load_params, res_param_names):

	# input assumed normalized(?) with shape (1,3,224,224)	

	X = T.ftensor4('input - X')
	# target assumed shape (1000,)
	Y = T.ivector('target - Y')
	LR = T.fscalar('learning rate')
	# load theano weights that aren't res params
	conv1_W = theano.shared(value=np.transpose(load_params['conv1/W'],(3,2,0,1)),borrow=True,name='conv1_W')
	bn1_conv1_gamma = theano.shared(load_params['bn_conv1/gamma'], borrow=True,name='bn_conv1_gamma')
	bn1_conv1_beta = theano.shared(load_params['bn_conv1/beta'], borrow=True,name='bn_conv1_beta')
	bn1_conv1_mean = theano.shared(load_params['bn_conv1/mean/EMA'],borrow=True,name='bn_conv1_mean')
	bn1_conv1_std = theano.shared(np.sqrt(load_params['bn_conv1/variance/EMA']),borrow=True,name='bn_conv1_std-dev')	

	conv1_params = [conv1_W,bn1_conv1_gamma,bn1_conv1_beta,bn1_conv1_mean,bn1_conv1_std]
	fc_W = theano.shared(load_params['fc1000/W'],borrow=True,name='fc1000_W')
	fc_b = theano.shared(load_params['fc1000/b'],borrow=True,name='fc1000_b')
	fc_params = [fc_W, fc_b]
	# build first convolution, batchnorm, and pool
	conv1_out = conv2d(input=X,
						filters=conv1_W,
						filter_shape=(64,3,7,7),
						subsample=(2,2),
						border_mode=(3,3))
	layer1_bn_out = T.nnet.relu(BN(inputs=conv1_out.dimshuffle(0,2,3,1),
								gamma=bn1_conv1_gamma,
								beta=bn1_conv1_beta, 
								mean=bn1_conv1_mean,
								std=bn1_conv1_std))
	# downsample of size 3 with stride of 2
	current_output = pool.pool_2d(input=layer1_bn_out.dimshuffle(0,3,1,2),
								ds=(3,3),
								st=(2,2),
								mode='max',
								ignore_border=False)
	current_dim = 64
	# build residual connections
	# 	first one no stride
	res_params = []
	current_output, current_params, current_dim = ResidualUnit(name=res_param_names[0],
								left_b=current_output,
								right_b=current_output,
								load_params=load_params,
								down_dim=64,
								up_dim=256,
								cur_dim=current_dim,
								stride=(1,1),
								left_convolve=True)
	res_params += current_params
	res_param_names = res_param_names[1:]
	for name in res_param_names:
		a = False
		st = (1,1)
		if name.endswith('a'):
			a = True
			down_dim = current_dim / 2
			up_dim = down_dim * 4
			st = (2,2)
		else:
			down_dim = current_dim / 4
			up_dim = down_dim * 4
		current_output, current_params, current_dim = ResidualUnit(name=name,
								left_b=current_output,
								right_b=current_output,
								load_params=load_params,
								down_dim=down_dim,
								up_dim=up_dim,
								cur_dim=current_dim,
								stride=st,
								left_convolve=a)
		res_params += current_params
	# pool before fc classifier
	pool2_out = pool.pool_2d(input=current_output,
							ds=(7,7),
							st=(1,1),
							mode='average_exc_pad',
							ignore_border=False)
	# final dense layer
	fc_out = T.dot(pool2_out.flatten(ndim=2),fc_W) + fc_b
	#fc_out = T.dot(fc_W,pool2_out) + fc_b	
	

	Y_hat = T.nnet.softmax(fc_out.flatten(ndim=2)) # prediction probabilities
	prediction = T.argmax(Y_hat) # prediction class 
	params = conv1_params + res_params + fc_params
	cost = T.nnet.categorical_crossentropy(Y_hat, Y).mean()
	grads = T.grad(cost,params)
	# dumb update... optimally sub a better one
	updates = [(weight, weight - LR*gradient) for weight, gradient in zip(params, grads)]
	train = theano.function(inputs=[X,Y],outputs=cost,updates=updates,givens={LR:lr})
	validate = theano.function(inputs=[X,Y],outputs=cost)
	predict = theano.function(inputs=[X], outputs=prediction)

	return train, validate, predict, params






print('loading ResNet101.npy file')
load_params = np.load('ResNet101.npy',encoding='latin1').item()


# this naming is taken from the way it is saved in the npy file
res_param_names = ['2a', '2b', '2c', '3a', '3b1', '3b2', '3b3', '4a', '4b1',
		 '4b2', '4b3', '4b4', '4b5', '4b6', '4b7', '4b8', '4b9','4b10', 
		 '4b11', '4b12', '4b13', '4b14', '4b15', '4b16', '4b17',
		 '4b18', '4b19', '4b20', '4b21', '4b22', '5a', '5b', '5c']





print('building model...')

train, validate, predict, params = build_model(load_params, res_param_names)























