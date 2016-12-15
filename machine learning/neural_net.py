'''
input > weight > hidden layer 1 (activation function) > weights > hidden layer 2 (activation function) > weights > output layer

compare output to intended output > cost function (cross entropy)

optimization function (optimizer) > minimize cost (AdamOptmizer ... SGD, AdaGrad)

backpropagation

feed forward + backprop = epoch
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('C:/Users/rodri/OneDrive/Documentos/python/extra/machine learning/', one_hot=True)

# One-Hot (one element is 'hot' the rest is colder off)
'''
Example 10 classes, 0-9
0 = [1,0,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0,0,0]
3 = [0,0,0,1,0,0,0,0,0,0]
4 = [0,0,0,0,1,0,0,0,0,0]
'''

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

# define placeholders and variables
# height x width
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
	# it will create a tensor ("array") of data using random numbers
	# ... (input data * weights) + biases
	# Advantage of using a biases: it will be useful in cases where the input is 0.
	hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])), 
						'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 
						'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
	
	hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), 
						'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])), 
						'biases':tf.Variable(tf.random_normal([n_classes]))}

	# (input data * weights) + biases

	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	# activation function
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

	return output

def train_neural_network(x):
	# 'One-Hot' format
	prediction = neural_network_model(x)
	# it will calculate the difference between the prediction we got to the no label that we have
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))

	# learning_rate = 0.001 (default)
	optmizer = tf.train.AdamOptimizer().minimize(cost)

	# cycles feed forward + backprop
	hm_epochs = 10

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		# training the network
		for epoch in range(hm_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				# optimize the cost passing the xs and ys
				_, c = sess.run([optmizer, cost], feed_dict={x: epoch_x, y: epoch_y})
				epoch_loss += c
			print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)