'''
input -> weight -> hidden layer 1 (activation function) -> 
weights -> hidden layer 2 (activation function) -> weights -> 
output layer

compare output to intended output -> cost function (cross entropy)
optimization function (optimizer) -> minimize cost (AdamOptimizer, ... , SGD, AdaGrad)

backpropagation

feed forward + back prop = epoch
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot = True)

'''
10 classes, 0-9

0 = [1,0,0,0,0,0,0,0,0,0]
1 = [0,1,0,0,0,0,0,0,0,0]
2 = [0,0,1,0,0,0,0,0,0,0]
3 = [0,0,0,1,0,0,0,0,0,0]
'''

nNodesHl1 = 500
nNodesHl2 = 500
nNodesHl3 = 500

nClasses = 10
batchSize = 10

# height X width
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neuralNetworkModel (data) :

    # (inputData * weights) + biases
    hidden1Layer = {'weights' : tf.Variable(tf.random_normal([784, nNodesHl1])),
                    'biases' : tf.Variable(tf.random_normal([nNodesHl1]))}

    hidden2Layer = {'weights' : tf.Variable(tf.random_normal([nNodesHl1, nNodesHl2])),
                    'biases' : tf.Variable(tf.random_normal([nNodesHl2]))}

    hidden3Layer = {'weights' : tf.Variable(tf.random_normal([nNodesHl2, nNodesHl3])),
                    'biases' : tf.Variable(tf.random_normal([nNodesHl3]))}

    outputLayer = {'weights' : tf.Variable(tf.random_normal([nNodesHl3, nClasses])),
                   'biases' : tf.Variable(tf.random_normal([nClasses]))}


    l1 = tf.matmul(data, hidden1Layer['weights']) + hidden1Layer['biases']
    l1 = tf.nn.relu(l1)

    l2 = tf.matmul(l1, hidden2Layer['weights']) + hidden2Layer['biases']
    l2 = tf.nn.relu(l2)

    l3 = tf.matmul(l2, hidden3Layer['weights']) + hidden3Layer['biases']
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, outputLayer['weights']) + outputLayer['biases']

    return output

def trainNeuralNetwork(x) :
    prediction = neuralNetworkModel(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))

    # learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # cycles feed forward + backprop
    noOfEpochs = 10

    with tf.Session() as sess :
        sess.run(tf.initialize_all_variables())

        for epoch in range(noOfEpochs) :
            epochLoss = 0

            for _ in range(int(mnist.train.noOfExamples / batchSize)) :
                epoch_x, epoch_y = mnist.train.next_batch(batchSize)
                _, c = sess.run([optimizer, cost], feed_dict = {x : epoch_x, y : epoch_y})

                epochLoss += c

            print('Epoch', epoch, 'completed out of', noOfEpochs,', loss :', epochLoss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy :', accuracy.eval({x : mnist.test.images, y : mnist.test.labels}))

trainNeuralNetwork(x)
