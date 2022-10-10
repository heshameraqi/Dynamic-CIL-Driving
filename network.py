"""
Adapted from:
https://github.com/carla-simulator/imitation-learning/blob/master/agents/imitation/imitation_learning_network.py
"""


from __future__ import print_function

import numpy as np

import tensorflow as tf


def weight_ones(shape, name):
    initial = tf.constant(1.0, shape=shape, name=name)
    return tf.Variable(initial)


def weight_xavi_init(shape, name):
    initial = tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())
    return initial


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)


class Network(object):

    def __init__(self, dropout, image_shape):
        """ We put a few counters to see how many times we called each function """
        self._dropout_vec = dropout
        self._image_shape = image_shape
        self._count_conv = 0
        self._count_pool = 0
        self._count_bn = 0
        self._count_activations = 0
        self._count_dropouts = 0
        self._count_fc = 0
        self._count_lstm = 0
        self._count_soft_max = 0
        self._conv_kernels = []
        self._conv_strides = []
        self._weights = {}
        self._features = {}

    """ Our conv is currently using bias """

    def conv(self, x, kernel_size, stride, output_size, padding_in='SAME'):
        self._count_conv += 1

        filters_in = x.get_shape()[-1]
        shape = [kernel_size, kernel_size, filters_in, output_size]

        weights = weight_xavi_init(shape, 'W_c_' + str(self._count_conv))
        bias = bias_variable([output_size], name='B_c_' + str(self._count_conv))

        self._weights['W_conv' + str(self._count_conv)] = weights
        self._conv_kernels.append(kernel_size)
        self._conv_strides.append(stride)

        conv_res = tf.add(tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding=padding_in,
                                       name='conv2d_' + str(self._count_conv)), bias,
                          name='add_' + str(self._count_conv))

        self._features['conv_block' + str(self._count_conv - 1)] = conv_res

        return conv_res

    def max_pool(self, x, ksize=3, stride=2):
        self._count_pool += 1
        return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, stride, stride, 1],
                              padding='SAME', name='max_pool' + str(self._count_pool))

    def bn(self, x, is_training):
        self._count_bn += 1
        return tf.contrib.layers.batch_norm(x, is_training=False,  # TODO: enable Batch Normalization
                                            updates_collections=None,
                                            scope='bn' + str(self._count_bn))

    def activation(self, x):
        self._count_activations += 1
        return tf.nn.relu(x, name='relu' + str(self._count_activations))

    def dropout(self, x):
        print("Dropout", self._count_dropouts)
        self._count_dropouts += 1
        output = tf.nn.dropout(x, self._dropout_vec[self._count_dropouts - 1],
                               name='dropout' + str(self._count_dropouts))

        return output

    def fc(self, x, output_size):
        self._count_fc += 1
        filters_in = x.get_shape()[-1]
        shape = [filters_in, output_size]

        weights = weight_xavi_init(shape, 'W_f_' + str(self._count_fc))
        bias = bias_variable([output_size], name='B_f_' + str(self._count_fc))

        return tf.nn.xw_plus_b(x, weights, bias, name='fc_' + str(self._count_fc))

    def conv_block(self, x, kernel_size, stride, output_size, is_training, padding_in='SAME'):
        print(" === Conv", self._count_conv, "  :  ", kernel_size, stride, output_size)
        with tf.name_scope("conv_block" + str(self._count_conv)):
            x = self.conv(x, kernel_size, stride, output_size, padding_in=padding_in)
            x = self.bn(x, is_training)
            x = self.dropout(x)  # Can be protected with an (if is_training) condition, but during deployment a dropout
                                 # vector of 1 is assumed give

            return self.activation(x)

    def fc_block(self, x, output_size, is_training):
        print(" === FC", self._count_fc, "  :  ", output_size)
        with tf.name_scope("fc" + str(self._count_fc + 1)):
            x = self.fc(x, output_size)
            x = self.dropout(x)  # Can be protected with an (if is_training) condition, but during deployment a dropout
                                 # vector of 1 is assumed give
            self._features['fc_block' + str(self._count_fc + 1)] = x
            return self.activation(x)

    def get_weigths_dict(self):
        return self._weights

    def get_feat_tensors_dict(self):
        return self._features

'''
model_inputs_mode should be either "1cam" or "1cam-pgm" or "3cams" or "3cams-pgm"
'''
def load_imitation_learning_network(model_inputs_mode, input_image, input_data,
                                    dropout_vector=[1.0] * 8 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.] * 5,
                                    branch_config=[["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"],
                                                   ["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"], ["Speed"]],
                                    is_training=True, input_image_right=None, input_image_left=None, input_image_lidar_pgm=None):
    if model_inputs_mode == "1cam":
        return load_network_1cam(input_image, input_data, dropout_vector=dropout_vector, branch_config=branch_config,
                          is_training=is_training)
    elif model_inputs_mode == "1cam-pgm":
        return load_network_1cam_pgm(input_image, input_image_lidar_pgm, input_data, dropout_vector=dropout_vector, branch_config=branch_config,
                          is_training=is_training)
    elif model_inputs_mode == "3cams":
        return load_network_3cams(input_image, input_image_right, input_image_left, input_data, dropout_vector=dropout_vector, branch_config=branch_config,
                           is_training=is_training)
    elif model_inputs_mode == "3cams-pgm":
        return load_network_3cams_pgm(input_image, input_image_right, input_image_left, input_image_lidar_pgm, input_data, dropout_vector=dropout_vector,
                                     branch_config=branch_config, is_training=is_training)
    else:
        assert "Error: model_inputs_mode is defined with unkown value, it should be either 1cam or 1cam-pgm or 3cams or 3cams-pgm"


def load_network_1cam(input_image, input_data,
                      dropout_vector=[1.0] * 8 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.] * 5,
                      branch_config=[["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"],
                                     ["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"], ["Speed"]],
                      is_training=True):
    branches = []

    x = input_image

    network_manager = Network(dropout_vector, tf.shape(x))

    """conv1"""  # kernel sz, stride, num feature maps
    print('Creating network ...')
    print(x)
    xc = network_manager.conv_block(x, 5, 2, 32, padding_in='VALID', is_training=is_training)
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 32, padding_in='VALID', is_training=is_training)
    print(xc)

    """conv2"""
    xc = network_manager.conv_block(xc, 3, 2, 64, padding_in='VALID', is_training=is_training)
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 64, padding_in='VALID', is_training=is_training)
    print(xc)

    """conv3"""
    xc = network_manager.conv_block(xc, 3, 2, 128, padding_in='VALID', is_training=is_training)
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 128, padding_in='VALID', is_training=is_training)
    print(xc)

    """conv4"""
    xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='VALID', is_training=is_training)
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='VALID', is_training=is_training)
    print(xc)
    """mp3 (default values)"""

    """ reshape """
    x = tf.reshape(xc, [-1, int(np.prod(xc.get_shape()[1:]))], name='reshape')
    print(x)

    """ fc1 """
    x = network_manager.fc_block(x, 512, is_training)
    print(x)
    """ fc2 """
    x = network_manager.fc_block(x, 512, is_training)

    """Process Control"""

    """ Speed (measurements)"""
    with tf.name_scope("Speed"):
        speed = input_data  # get the speed from input data
        speed = network_manager.fc_block(speed, 128, is_training)
        speed = network_manager.fc_block(speed, 128, is_training)

    """ Joint sensory """
    j = tf.concat([x, speed], 1)
    j = network_manager.fc_block(j, 512, is_training)

    """Start BRANCHING according to branch_config """
    for i in range(0, len(branch_config)):
        with tf.name_scope("Branch_" + str(i)):
            if branch_config[i][0] == "Speed":
                # we only use the image as input to speed prediction
                branch_output = network_manager.fc_block(x, 256, is_training)
                branch_output = network_manager.fc_block(branch_output, 256, is_training)
            else:
                branch_output = network_manager.fc_block(j, 256, is_training)
                branch_output = network_manager.fc_block(branch_output, 256, is_training)

            branches.append(network_manager.fc(branch_output, len(branch_config[i])))

        print(branch_output)

    return branches

def load_network_1cam_pgm(input_image, input_image_lidar_pgm, input_data,
                                 dropout_vector=[1.0] * 8 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.] * 5,
                                 branch_config=[["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"],
                                                ["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"],
                                                ["Speed"]], is_training=True):
    branches = []

    # First image stream
    x = input_image

    network_manager = Network(dropout_vector, tf.shape(x))

    """conv1"""  # kernel sz, stride, num feature maps
    print('Creating network ...')
    print("First input stream ...")
    print(x)
    xc = network_manager.conv_block(x, 5, 2, 32, padding_in='VALID', is_training=is_training)
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 32, padding_in='VALID', is_training=is_training)
    print(xc)

    """conv2"""
    xc = network_manager.conv_block(xc, 3, 2, 64, padding_in='VALID', is_training=is_training)
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 64, padding_in='VALID', is_training=is_training)
    print(xc)

    """conv3"""
    xc = network_manager.conv_block(xc, 3, 2, 128, padding_in='VALID', is_training=is_training)
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 128, padding_in='VALID', is_training=is_training)
    print(xc)

    """conv4"""
    xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='VALID', is_training=is_training)
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='VALID', is_training=is_training)
    print(xc)
    """mp3 (default values)"""

    """ reshape """
    x1 = tf.reshape(xc, [-1, int(np.prod(xc.get_shape()[1:]))], name='reshape')
    print(x1)
        
    # Second image stream
    print("Fourth input stream ...")
    x = input_image_lidar_pgm
    x = tf.expand_dims(x,axis=-1)  # Because placeholder is 2D for LiDAR PGM

    """conv1"""  # kernel sz, stride, num feature maps
    print(x)
    xc = network_manager.conv_block(x, 5, 2, 32, padding_in='VALID', is_training=is_training)
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 32, padding_in='VALID', is_training=is_training)
    print(xc)

    """conv2"""
    xc = network_manager.conv_block(xc, 3, 2, 64, padding_in='VALID', is_training=is_training)
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 64, padding_in='VALID', is_training=is_training)
    print(xc)

    '''"""conv3"""
    xc = network_manager.conv_block(xc, 3, 2, 128, padding_in='VALID', is_training=is_training)
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 128, padding_in='VALID', is_training=is_training)
    print(xc)

    """conv4"""
    xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='VALID', is_training=is_training)
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='VALID', is_training=is_training)
    print(xc)
    """mp3 (default values)"""'''

    """ reshape """
    x2 = tf.reshape(xc, [-1, int(np.prod(xc.get_shape()[1:]))], name='reshape')
    print(x2)

    """ fc1 """
    x = tf.concat([x1, x2], 1)
    x = network_manager.fc_block(x, 512, is_training)
    print(x)
    """ fc2 """
    x = network_manager.fc_block(x, 512, is_training)

    """Process Control"""

    """ Speed (measurements)"""
    with tf.name_scope("Speed"):
        speed = input_data  # get the speed from input data
        speed = network_manager.fc_block(speed, 128, is_training)
        speed = network_manager.fc_block(speed, 128, is_training)

    """ Joint sensory """
    j = tf.concat([x, speed], 1)
    j = network_manager.fc_block(j, 512, is_training)

    """Start BRANCHING according to branch_config """
    for i in range(0, len(branch_config)):
        with tf.name_scope("Branch_" + str(i)):
            if branch_config[i][0] == "Speed":
                # we only use the image as input to speed prediction
                branch_output = network_manager.fc_block(x, 256, is_training)
                branch_output = network_manager.fc_block(branch_output, 256, is_training)
            else:
                branch_output = network_manager.fc_block(j, 256, is_training)
                branch_output = network_manager.fc_block(branch_output, 256, is_training)

            branches.append(network_manager.fc(branch_output, len(branch_config[i])))

        print(branch_output)

    return branches    
    
def load_network_3cams(input_image, input_image_right, input_image_left, input_data,
                       dropout_vector=[1.0] * 8 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.] * 5,
                       branch_config=[["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"],
                                      ["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"],
                                      ["Speed"]], is_training=True):
    branches = []

    # First image stream
    x = input_image

    network_manager = Network(dropout_vector, tf.shape(x))

    """conv1"""  # kernel sz, stride, num feature maps
    print('Creating network ...')
    print("First input stream ...")
    print(x)
    xc = network_manager.conv_block(x, 5, 2, 32, padding_in='VALID', is_training=is_training)
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 32, padding_in='VALID', is_training=is_training)
    print(xc)

    """conv2"""
    xc = network_manager.conv_block(xc, 3, 2, 64, padding_in='VALID', is_training=is_training)
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 64, padding_in='VALID', is_training=is_training)
    print(xc)

    """conv3"""
    xc = network_manager.conv_block(xc, 3, 2, 128, padding_in='VALID', is_training=is_training)
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 128, padding_in='VALID', is_training=is_training)
    print(xc)

    """conv4"""
    xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='VALID', is_training=is_training)
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='VALID', is_training=is_training)
    print(xc)
    """mp3 (default values)"""

    """ reshape """
    x1 = tf.reshape(xc, [-1, int(np.prod(xc.get_shape()[1:]))], name='reshape')
    print(x1)
    
    # Second image stream
    print("Second input stream ...")
    x = input_image_right

    """conv1"""  # kernel sz, stride, num feature maps
    print(x)
    xc = network_manager.conv_block(x, 5, 2, 32, padding_in='VALID', is_training=is_training)
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 32, padding_in='VALID', is_training=is_training)
    print(xc)

    """conv2"""
    xc = network_manager.conv_block(xc, 3, 2, 64, padding_in='VALID', is_training=is_training)
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 64, padding_in='VALID', is_training=is_training)
    print(xc)

    """conv3"""
    xc = network_manager.conv_block(xc, 3, 2, 128, padding_in='VALID', is_training=is_training)
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 128, padding_in='VALID', is_training=is_training)
    print(xc)

    """conv4"""
    xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='VALID', is_training=is_training)
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='VALID', is_training=is_training)
    print(xc)
    """mp3 (default values)"""

    """ reshape """
    x2 = tf.reshape(xc, [-1, int(np.prod(xc.get_shape()[1:]))], name='reshape')
    print(x2)
    
    # Third image stream
    print("Third input stream ...")
    x = input_image_left

    """conv1"""  # kernel sz, stride, num feature maps
    print(x)
    xc = network_manager.conv_block(x, 5, 2, 32, padding_in='VALID', is_training=is_training)
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 32, padding_in='VALID', is_training=is_training)
    print(xc)

    """conv2"""
    xc = network_manager.conv_block(xc, 3, 2, 64, padding_in='VALID', is_training=is_training)
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 64, padding_in='VALID', is_training=is_training)
    print(xc)

    """conv3"""
    xc = network_manager.conv_block(xc, 3, 2, 128, padding_in='VALID', is_training=is_training)
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 128, padding_in='VALID', is_training=is_training)
    print(xc)

    """conv4"""
    xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='VALID', is_training=is_training)
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='VALID', is_training=is_training)
    print(xc)
    """mp3 (default values)"""

    """ reshape """
    x3 = tf.reshape(xc, [-1, int(np.prod(xc.get_shape()[1:]))], name='reshape')
    print(x3)

    """ fc1 """
    x = tf.concat([x1, x2, x3], 1)
    x = network_manager.fc_block(x, 512, is_training)
    print(x)
    """ fc2 """
    x = network_manager.fc_block(x, 512, is_training)

    """Process Control"""

    """ Speed (measurements)"""
    with tf.name_scope("Speed"):
        speed = input_data  # get the speed from input data
        speed = network_manager.fc_block(speed, 128, is_training)
        speed = network_manager.fc_block(speed, 128, is_training)

    """ Joint sensory """
    j = tf.concat([x, speed], 1)
    j = network_manager.fc_block(j, 512, is_training)

    """Start BRANCHING according to branch_config """
    for i in range(0, len(branch_config)):
        with tf.name_scope("Branch_" + str(i)):
            if branch_config[i][0] == "Speed":
                # we only use the image as input to speed prediction
                branch_output = network_manager.fc_block(x, 256, is_training)
                branch_output = network_manager.fc_block(branch_output, 256, is_training)
            else:
                branch_output = network_manager.fc_block(j, 256, is_training)
                branch_output = network_manager.fc_block(branch_output, 256, is_training)

            branches.append(network_manager.fc(branch_output, len(branch_config[i])))

        print(branch_output)

    return branches

def load_network_3cams_pgm(input_image, input_image_right, input_image_left, input_image_lidar_pgm, input_data,
                                 dropout_vector=[1.0] * 8 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.] * 5,
                                 branch_config=[["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"],
                                                ["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"],
                                                ["Speed"]], is_training=True):
    branches = []

    # First image stream
    x = input_image

    network_manager = Network(dropout_vector, tf.shape(x))

    """conv1"""  # kernel sz, stride, num feature maps
    print('Creating network ...')
    print("First input stream ...")
    print(x)
    xc = network_manager.conv_block(x, 5, 2, 32, padding_in='VALID', is_training=is_training)
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 32, padding_in='VALID', is_training=is_training)
    print(xc)

    """conv2"""
    xc = network_manager.conv_block(xc, 3, 2, 64, padding_in='VALID', is_training=is_training)
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 64, padding_in='VALID', is_training=is_training)
    print(xc)

    """conv3"""
    xc = network_manager.conv_block(xc, 3, 2, 128, padding_in='VALID', is_training=is_training)
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 128, padding_in='VALID', is_training=is_training)
    print(xc)

    """conv4"""
    xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='VALID', is_training=is_training)
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='VALID', is_training=is_training)
    print(xc)
    """mp3 (default values)"""

    """ reshape """
    x1 = tf.reshape(xc, [-1, int(np.prod(xc.get_shape()[1:]))], name='reshape')
    print(x1)
    
    # Second image stream
    print("Second input stream ...")
    x = input_image_right

    """conv1"""  # kernel sz, stride, num feature maps
    print(x)
    xc = network_manager.conv_block(x, 5, 2, 32, padding_in='VALID', is_training=is_training)
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 32, padding_in='VALID', is_training=is_training)
    print(xc)

    """conv2"""
    xc = network_manager.conv_block(xc, 3, 2, 64, padding_in='VALID', is_training=is_training)
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 64, padding_in='VALID', is_training=is_training)
    print(xc)

    """conv3"""
    xc = network_manager.conv_block(xc, 3, 2, 128, padding_in='VALID', is_training=is_training)
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 128, padding_in='VALID', is_training=is_training)
    print(xc)

    """conv4"""
    xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='VALID', is_training=is_training)
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='VALID', is_training=is_training)
    print(xc)
    """mp3 (default values)"""

    """ reshape """
    x2 = tf.reshape(xc, [-1, int(np.prod(xc.get_shape()[1:]))], name='reshape')
    print(x2)
    
    # Third image stream
    print("Third input stream ...")
    x = input_image_left

    """conv1"""  # kernel sz, stride, num feature maps
    print(x)
    xc = network_manager.conv_block(x, 5, 2, 32, padding_in='VALID', is_training=is_training)
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 32, padding_in='VALID', is_training=is_training)
    print(xc)

    """conv2"""
    xc = network_manager.conv_block(xc, 3, 2, 64, padding_in='VALID', is_training=is_training)
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 64, padding_in='VALID', is_training=is_training)
    print(xc)

    """conv3"""
    xc = network_manager.conv_block(xc, 3, 2, 128, padding_in='VALID', is_training=is_training)
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 128, padding_in='VALID', is_training=is_training)
    print(xc)

    """conv4"""
    xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='VALID', is_training=is_training)
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='VALID', is_training=is_training)
    print(xc)
    """mp3 (default values)"""

    """ reshape """
    x3 = tf.reshape(xc, [-1, int(np.prod(xc.get_shape()[1:]))], name='reshape')
    print(x3)
    
    # Fourth image stream
    print("Fourth input stream ...")
    x = input_image_lidar_pgm
    x = tf.expand_dims(x,axis=-1)  # Because placeholder is 2D for LiDAR PGM

    """conv1"""  # kernel sz, stride, num feature maps
    print(x)
    xc = network_manager.conv_block(x, 5, 2, 32, padding_in='VALID', is_training=is_training)
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 32, padding_in='VALID', is_training=is_training)
    print(xc)

    """conv2"""
    xc = network_manager.conv_block(xc, 3, 2, 64, padding_in='VALID', is_training=is_training)
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 64, padding_in='VALID', is_training=is_training)
    print(xc)

    '''"""conv3"""
    xc = network_manager.conv_block(xc, 3, 2, 128, padding_in='VALID', is_training=is_training)
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 128, padding_in='VALID', is_training=is_training)
    print(xc)

    """conv4"""
    xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='VALID', is_training=is_training)
    print(xc)
    xc = network_manager.conv_block(xc, 3, 1, 256, padding_in='VALID', is_training=is_training)
    print(xc)
    """mp3 (default values)"""'''

    """ reshape """
    x4 = tf.reshape(xc, [-1, int(np.prod(xc.get_shape()[1:]))], name='reshape')
    print(x4)

    """ fc1 """
    x = tf.concat([x1, x2, x3, x4], 1)
    x = network_manager.fc_block(x, 512, is_training)
    print(x)
    """ fc2 """
    x = network_manager.fc_block(x, 512, is_training)

    """Process Control"""

    """ Speed (measurements)"""
    with tf.name_scope("Speed"):
        speed = input_data  # get the speed from input data
        speed = network_manager.fc_block(speed, 128, is_training)
        speed = network_manager.fc_block(speed, 128, is_training)

    """ Joint sensory """
    j = tf.concat([x, speed], 1)
    j = network_manager.fc_block(j, 512, is_training)

    """Start BRANCHING according to branch_config """
    for i in range(0, len(branch_config)):
        with tf.name_scope("Branch_" + str(i)):
            if branch_config[i][0] == "Speed":
                # we only use the image as input to speed prediction
                branch_output = network_manager.fc_block(x, 256, is_training)
                branch_output = network_manager.fc_block(branch_output, 256, is_training)
            else:
                branch_output = network_manager.fc_block(j, 256, is_training)
                branch_output = network_manager.fc_block(branch_output, 256, is_training)

            branches.append(network_manager.fc(branch_output, len(branch_config[i])))

        print(branch_output)

    return branches