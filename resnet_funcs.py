
import numpy as np 
import theano
from theano import tensor as T 
from theano import config
from theano.tensor.nnet import conv2d
from theano.tensor.nnet.bn import batch_normalization as BN


#load_params = np.load('ResNet101.npy',encoding='latin1').item()
#X = T.tensor4()
#n = '2a'


def ResidualUnit(name,left_b,right_b,load_params,cur_dim,down_dim,up_dim,stride=(1,1),left_convolve=False):
	# param key names
	left_res = 'res%s_branch1' % (name)
	left_bn = 'bn%s_branch1' % (name)

	right_res_a = 'res%s_branch2a' % (name)
	right_bn_a = 'bn%s_branch2a' % (name)
	right_res_b = 'res%s_branch2b' % (name)
	right_bn_b = 'bn%s_branch2b' % (name)
	right_res_c = 'res%s_branch2c' % (name)
	right_bn_c = 'bn%s_branch2c' % (name)

	# init theano weights
	# NOTE: the np.loaded params do not have any bias vectors 
	#     except for the final dense layer, so no bias additions here
	params = []
	if left_convolve:
		left_res_W = theano.shared(value=np.transpose(load_params[left_res+'/W'],(3,2,0,1)),borrow=True,name=left_res+'_W')
		left_bn_beta = theano.shared(load_params[left_bn+'/beta'],borrow=True,name=left_bn+'_beta')
		left_bn_gamma = theano.shared(load_params[left_bn+'/gamma'],borrow=True,name=left_bn+'_gamma')
		left_bn_mean = theano.shared(load_params[left_bn+'/mean/EMA'],borrow=True,name=left_bn+'_mean')
		left_bn_std = theano.shared(np.sqrt(load_params[left_bn+'/variance/EMA']),borrow=True,name=left_bn+'_std-dev')
		params += [left_res_W, left_bn_gamma,left_bn_beta,left_bn_mean,left_bn_std]
	
	right_res_a_W = theano.shared(value=np.transpose(load_params[right_res_a+'/W'],(3,2,0,1)),borrow=True,name=right_res_a+'_W')
	right_bn_a_beta = theano.shared(load_params[right_bn_a+'/beta'],borrow=True,name=right_bn_a+'_beta')	
	right_bn_a_gamma = theano.shared(load_params[right_bn_a+'/gamma'],borrow=True,name=right_bn_a+'_gamma')
	right_bn_a_mean = theano.shared(load_params[right_bn_a+'/mean/EMA'],borrow=True,name=right_bn_a+'_mean')
	right_bn_a_std = theano.shared(np.sqrt(load_params[right_bn_a+'/variance/EMA']),borrow=True,name=right_bn_a+'_std-dev')

	right_res_b_W = theano.shared(value=np.transpose(load_params[right_res_b+'/W'],(3,2,0,1)),borrow=True,name=right_res_b+'_W')
	right_bn_b_beta = theano.shared(load_params[right_bn_b+'/beta'],borrow=True,name=right_bn_b+'_beta')	
	right_bn_b_gamma = theano.shared(load_params[right_bn_b+'/gamma'],borrow=True,name=right_bn_b+'_gamma')
	right_bn_b_mean = theano.shared(load_params[right_bn_b+'/mean/EMA'],borrow=True,name=right_bn_b+'_mean')
	right_bn_b_std = theano.shared(np.sqrt(load_params[right_bn_b+'/variance/EMA']),borrow=True,name=right_bn_b+'_std-dev')

	right_res_c_W = theano.shared(value=np.transpose(load_params[right_res_c+'/W'],(3,2,0,1)),borrow=True,name=right_res_c+'_W')
	right_bn_c_beta = theano.shared(load_params[right_bn_c+'/beta'],borrow=True,name=right_bn_c+'_beta')	
	right_bn_c_gamma = theano.shared(load_params[right_bn_c+'/gamma'],borrow=True,name=right_bn_c+'_gamma')
	right_bn_c_mean = theano.shared(load_params[right_bn_c+'/mean/EMA'],borrow=True,name=right_bn_c+'_mean')
	right_bn_c_std = theano.shared(np.sqrt(load_params[right_bn_c+'/variance/EMA']),borrow=True,name=right_bn_c+'_std-dev')

	params += [right_res_a_W,right_bn_a_gamma,right_bn_a_beta,right_bn_a_mean,right_bn_a_std,
				right_res_b_W,right_bn_b_gamma,right_bn_b_beta,right_bn_b_mean,right_bn_b_std,
				right_res_c_W,right_bn_c_gamma,right_bn_c_beta,right_bn_c_mean,right_bn_c_std]
	

	# make tensor graph
	if left_convolve:
		left_conv_out = conv2d(input=left_b,filters=left_res_W,filter_shape=(up_dim,cur_dim,1,1),subsample=stride)
		left_out = BN(inputs=left_conv_out.dimshuffle(0,2,3,1),gamma=left_bn_gamma,beta=left_bn_beta,
					mean=left_bn_mean,std=left_bn_std).dimshuffle(0,3,1,2)
	else:
		left_out = left_b

	right_conv_a_out = conv2d(input=right_b,filters=right_res_a_W,filter_shape=(down_dim,cur_dim,1,1),subsample=stride)
	right_a_out = T.nnet.relu(BN(inputs=right_conv_a_out.dimshuffle(0,2,3,1),gamma=right_bn_a_gamma,beta=right_bn_a_beta,
						mean=right_bn_a_mean,std=right_bn_a_std).dimshuffle(0,3,1,2))
	right_conv_b_out = conv2d(input=right_a_out,filters=right_res_b_W,filter_shape=(down_dim,down_dim,3,3),border_mode=(1,1))
	right_b_out = T.nnet.relu(BN(inputs=right_conv_b_out.dimshuffle(0,2,3,1),gamma=right_bn_b_gamma,beta=right_bn_b_beta,
						mean=right_bn_b_mean,std=right_bn_b_std).dimshuffle(0,3,1,2))
	right_c_conv_out = conv2d(input=right_b_out,filters=right_res_c_W,filter_shape=(up_dim,down_dim,1,1))
	right_out = BN(inputs=right_c_conv_out.dimshuffle(0,2,3,1),gamma=right_bn_c_gamma,beta=right_bn_c_beta,
						mean=right_bn_c_mean,std=right_bn_c_std).dimshuffle(0,3,1,2)

	output = T.nnet.relu(left_out + right_out)

	return output, params, 4*down_dim















