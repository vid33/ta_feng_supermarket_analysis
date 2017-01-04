import pandas as pd
import numpy as np
import tensorflow as tf

from imp import reload

"""
PLAN: Do some supervised learning w.r.t. age and residence area. Start with log. reg./softmax, go on to
more complex stuff (thus using tensorflow). Idea is to see which features are responsible for various
age groups. Might be of interest, e.g. 35-40 age group is responsible for a very large fraction of sales.
Can also use this to fill in missing values.
"""

"""
NOTE: At the moment this is just softmax, trained on one month, and tested on another.
Working file, and a little messy.
"""

# Classifiers need to be in [000100...] format
def tensorflow_format(age_list, age_data):

	age_data_tf = np.zeros([len(age_data), len(age_list)])
	age_dict = {}

	for i, age in enumerate(age_list):
		tmp_zeros = np.zeros(len(age_list))
		tmp_zeros[i] = 1
		age_dict[age_list[i]] = tmp_zeros
	
	for i, age_datum in np.ndenumerate(age_data):
		age_data_tf[i, :] = age_dict[age_datum]
			
	return age_data_tf, age_dict

# Binary classifier, age is/is not e.g. 37
def tensorflow_format_binary(age_list, age_data):

	age_data_tf = np.zeros([len(age_data), 2])
	age_dict = {}

	for i, age in enumerate(age_list):
		tmp_zeros = np.zeros(2)
		if age == 37: # Classify: equals, or does not equal, this age
			tmp_zeros[0] = 1
		else:
			tmp_zeros[1] = 1
		age_dict[age_list[i]] = tmp_zeros
	
	for i, age_datum in np.ndenumerate(age_data):
		age_data_tf[i, :] = age_dict[age_datum]
			
	return age_data_tf, age_dict

def testing_accuracy(data_period, W, bb):

	print('Testing using:', data_period)

	features_file = 'data/' + 'features_array_' + data_period + '.npy'
	print('Loading testing features:', features_file)
	features=np.load(features_file)

	age_data = features[:,0]
	age_list=  np.unique(age_data)  

	features_testing = features[:, 1:]
	features_testing = (features_testing - np.mean(features_testing, axis=0))/np.var(features_testing, axis=0)


	# Put classifiers into [000100...] format
	#age_training_tf, age_dict = tensorflow_format(age_list, age_data)
	# Or binary classifier: is or is not equal to some specified age
	age_testing, age_dict = tensorflow_format_binary(age_list, age_data)

	age_out = features_testing.dot(W) + bb

	correct_prediction = np.equal(np.argmax(age_out, axis=1), np.argmax(age_testing, axis=1))

	return correct_prediction


def softmax_age():

	"""  Meant to predict user age from features. 
	NOTE: Works OK in binary form, i.e. logistic, 80% testing accuracy, and badly if all
	age groups included. Overall very basic code, and some things are a little suspect""" 

	# Pick one of 'nov_00', 'dec_00', 'jan_01', 'feb_01', 'all', or 'all_periodic'
	data_period= 'nov_00'
	print('Data set:', data_period)

	features_file = 'data/' + 'features_array_' + data_period + '.npy'
	print('Loading features:', features_file)
	features=np.load(features_file)

	age_data = features[:,0]
	age_list=  np.unique(age_data)  

	features_training_tf = features[:, 1:]

	#features_training_tf = np.log(features_training_tf)  #does nothing for softmax.

	#Normalise data
	features_training_tf = (features_training_tf - np.mean(features_training_tf, axis=0))/np.var(features_training_tf, axis=0)

	# Put classifiers into [000100...] format
	#age_training_tf, age_dict = tensorflow_format(age_list, age_data)
	# Or binary classifier: is or is not equal to some specified age
	age_training_tf, age_dict = tensorflow_format_binary(age_list, age_data)

	feature_no = features_training_tf.shape[1]
	class_no =  age_training_tf.shape[1]

	# Tensorflow model
	x = tf.placeholder(tf.float32, [None, feature_no])
	W_loc = tf.Variable(tf.zeros([feature_no, class_no]))
	b_loc = tf.Variable(tf.zeros([class_no]))
	y = tf.matmul(x, W_loc) + b_loc

	y_ = tf.placeholder(tf.float32, [None, class_no])

	# Defines the cost function, cross entropy as we are doing softmax
	learning_rate = 0.5
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

	sess = tf.InteractiveSession()

	tf.initialize_all_variables().run()

	training_steps = 500
	for _ in range(training_steps):
		batch_xs, batch_ys = features_training_tf, age_training_tf # Currently slow, loads the whole training set
		xs_glob = batch_xs
		ys_glob = batch_ys
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

	W = sess.run(W_loc)
	bb = sess.run(b_loc)

	y_trained = features_training_tf.dot(W) + bb
	y_true = age_training_tf;
	correct_prediction = np.equal(np.argmax(y_trained, axis=1), np.argmax(y_true, axis=1))

	# Tensorflow way of calculating the above
	"""
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print("Training accuracy", sess.run(accuracy, feed_dict={x: features_training_tf, y_: age_training_tf}))
	"""
	print("Training accuracy", sum(correct_prediction)/len(correct_prediction))
	correct_prediction_testing = testing_accuracy('feb_01', W, bb)
	print("Testing accuracy", sum(correct_prediction_testing)/len(correct_prediction_testing))

if __name__ == "__main__":
    softmax_age()

