"""
===============================================================================
Convolutional Neural Network
===============================================================================
author=hal112358
-------------------------------------------------------------------------------
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np
import tensorboard 
import sys
import os


class convolutional_neural_network():

    def __init__(self,inputs,labels):
        self.name            = 'BreakHis'
        
        self.inputs          = inputs
        self.labels          = labels
        
        # Integers
        self.n_input         = len(self.inputs[0])
        self.n_classes       = len(self.labels[0])
        self.training_iters  = 200000
        self.evaluation_c    = 256
        self.batch_size      = 256
        self.evaluation_c    = 256
        self.epochs          = 50
        self.display_step    = 1
        
        # Floats
        self.learning_rate   = 0.001
        self.dropout         = 0.75
        
        # Booleans
        self.save            = True
        
        # Tf objects
        self.x               = tf.placeholder(tf.float32,[None,self.n_input])
        self.y               = tf.placeholder(tf.float32,[None,self.n_classes])
        self.keep_prob       = tf.placeholder(tf.float32)
        
        randVar = lambda shape: tf.Variable(tf.random_normal(shape))
        
        self.weights         = {
            'wc1'            :   randVar([5,5,1,32]),
            'wc2'            :   randVar([5,5,32,64]),
            'wf1'            :   randVar([7*7*64,1024]),
            'wf2'            :   randVar([1024,1024]),
            'wf3'            :   randVar([1024,self.n_classes])
        }
        
        self.biases          = {
            'bc1'            :   randVar([32]),
            'bc2'            :   randVar([64]),
            'bf1'            :   randVar([1024]),
            'bf2'            :   randVar([1024]),
            'bf3'            :   randVar([self.n_classes])
        }
        
        # Paths

        self.model_folder    = os.path.join(folder_paths['saved_models'],
                                'models/')
        self.model_path      = os.path.join(self.model_folder,self.name)

    """
    ===========================================================================
    Tensorflow Layers
    ===========================================================================
    """

    def conv2d(self,x,W,b,strides=1):
        """
        Tf 2D convolutional layer
        """
        return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(
            x,W,strides=[1,strides,strides,1],padding="SAME"),b))

    #--------------------------------------------------------------------------

    def maxpool2d(self,x,k=2):
        return tf.nn.max_pool(x,ksize=[1,k,k,1],strides=[1,k,k,1],
                padding="SAME")

    #--------------------------------------------------------------------------

    def conv_maxpool_layer(self,input_data,index):
        current_layer = self.maxpool2d(self.conv2d(input_data,
            self.weights['wc{}'.format(index)],
            self.biases['bc{}'.format(index)]))
        tf.summary.histogram("conv_maxpool_layer_{}".format(index),current_layer)
        return current_layer
        
    #--------------------------------------------------------------------------

    def fully_connected_layer(self,input_layer,index,dropout):
        input_layer = tf.reshape(input_layer, [-1, 
                self.weights['wf{}'.format(index)].get_shape().as_list()[0]])
    
        current_layer = tf.add(tf.matmul(input_layer,
                self.weights["wf{}".format(index)]),
                self.biases['bf{}'.format(index)])
        
        current_layer = tf.nn.relu(tf.nn.dropout(current_layer, dropout))
        tf.summary.histogram("feed_forward_layer_{}".format(index),
            current_layer)
        
        return current_layer

    #--------------------------------------------------------------------------

    def network(self,x,weights,biases,dropout):
        x     = tf.reshape(self.x,shape=[-1,int(np.sqrt(self.n_input)),
                    int(np.sqrt(self.n_input)),1])
        conv1 = self.conv_maxpool_layer(x,1)
        conv2 = self.conv_maxpool_layer(conv1,2)
        fc1   = self.fully_connected_layer(conv2,1,dropout)
        fc2   = self.fully_connected_layer(fc1,2,dropout)
        fc3   = tf.add(tf.matmul(fc2, self.weights['wf3']), self.biases['bf3'])
        return fc3

    """
    ===========================================================================
    Network Training
    ===========================================================================
    """

    def display(self,epoch,step,loss,accuracy):
        self.log_info("Epoch: {}  Iter:{}  Loss:{}  Accuracy:{}".format(epoch,
            step*self.batch_size,round(loss,3),round(accuracy,7)))

    #--------------------------------------------------------------------------

    def test_evaluation(self,sess):
        """
        Test overall accuracy against testing set that was not used 
        during training
        """
        test_acc = sess.run(self.accuracy, 
                    feed_dict={self.x:self.inputs[:self.evaluation_c],
                               self.y:self.labels[:self.evaluation_c],
                               self.keep_prob:1
                               })
        self.log_info("Testing Accuracy: {}".format(test_acc))

    #--------------------------------------------------------------------------
        
    def initialize_tensorBoard(self):
        """
        Setup the variables that will be used in Tensorboard to view the 
        network training performance. 
        """
        regularization_losses   = tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES)

        regularization_constant = 0.01
        
        with tf.name_scope('Model'):
            prediction = self.network(self.x, self.weights,
                self.biases, self.keep_prob)
            
        with tf.name_scope('Cost'):
            self.cost = (tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(
                        logits=prediction,targets=self.y,pos_weight=0.1))+
                        (regularization_constant*sum(regularization_losses)))
            
        with tf.name_scope('Optimizer'):
            self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate).minimize(self.cost)
            
        tf.summary.scalar("Cost",self.cost)
        
        self.merged_summary_op = tf.summary.merge_all()

        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #--------------------------------------------------------------------------

    def train(self):
        self.initialize_tensorBoard()
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            summary_writer = tf.summary.FileWriter(self.logs_path,
                                graph=tf.get_default_graph())
            saver = tf.train.Saver()
            
            for epoch in range(self.epochs):
                step = 1
                continue_batch = (step*self.batch_size < self.training_iters)
                fit_batch      = (step*self.batch_size<len(self.inputs))
                while continue_batch and fit_batch:
                    batch_x = self.inputs[(step-1)*self.batch_size:step*self.batch_size]
                    batch_y = self.labels[(step-1)*self.batch_size:step*self.batch_size]
                    
                    _, __, summary = sess.run([self.optimizer,self.cost,
                                     self.merged_summary_op],
                                     feed_dict={self.x         : batch_x,
                                                self.y         : batch_y,
                                                self.keep_prob : self.dropout
                                                })
                    
                    summary_writer.add_summary(summary, epoch*self.batch_size+step)
                    
                    if not step % self.display_step:
                        loss, accuracy = sess.run([self.cost, self.accuracy],
                                         feed_dict={
                                          self.x         : batch_x, 
                                          self.y         : batch_y,
                                          self.keep_prob : 1.
                                         })
    
                        self.display(epoch+1,step,loss,accuracy)
                    step += 1
                    
            if self.save:
                if not os.path.exists(self.model_folder):
                    os.mkdir(self.model_folder)
                    
                saver.save(sess,"{}.ckpt".format(self.model_path),global_step=1000)
                
            self.test_evaluation(sess)
            os.system("tensorboard --logdir={}".format(self.logs_path))

"""
===============================================================================
Network Testing (MNIST)
===============================================================================
"""

def MNIST_test():
    from tensorflow.examples.tutorials.mnist import input_data
    n_train = 5500
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    inputs, labels = mnist.train.next_batch(mnist.train.num_examples)
    training_inputs, training_labels = inputs[:n_train],labels[:n_train]
    testing_inputs, testing_labels = inputs[n_train:],labels[n_train:]
    test_datapoint = np.reshape(testing_inputs[0],(1,-1))
    
    MNIST_net = SkyNet_CNN(training_inputs,training_labels)
    MNIST_net.train_main()