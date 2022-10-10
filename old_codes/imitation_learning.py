from __future__ import print_function

import os

import scipy

import tensorflow as tf
import numpy as np

from carla.agent import Agent
from carla.carla_server_pb2 import Control
from network import load_imitation_learning_network


class ImitationLearning(Agent):
    '''load_ready_model: because ready model assumed a different order of network output branches'''
    def __init__(self, model_inputs_mode, model_path, avoid_stopping, memory_fraction=0.25,
                 image_cut=[115, 510], load_ready_model=False, gpu=0):

        Agent.__init__(self)

        self._image_cut = image_cut
        self.dropout_vec = [1.0] * 8 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.] * 5

        # Seen GPU's and memory_fraction used are selected
        #tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.visible_device_list = str(gpu)
        # tf_config.gpu_options.per_process_gpu_memory_fraction = memory_fraction
        # tf.reset_default_graph()
        sessGraph = tf.Graph()
    
        self._image_size = (88, 200, 3)
        self._avoid_stopping = avoid_stopping
        self._load_ready_model = load_ready_model

        with tf.device('/gpu:' + str(gpu)):
            with sessGraph.as_default():
                sess = tf.Session(graph=sessGraph, config=tf_config)
                with sess.as_default():
                    self.sess = sess
                    self._input_images = tf.placeholder("float", shape=[None, self._image_size[0],
                                                                        self._image_size[1],
                                                                        self._image_size[2]],
                                                        name="input_image")

                    self._input_data = []

                    self._input_data.append(tf.placeholder(tf.float32,
                                                           shape=[None, 4], name="input_control"))

                    self._input_data.append(tf.placeholder(tf.float32,
                                                           shape=[None, 1], name="input_speed"))

                    self._dout = tf.placeholder("float", shape=[len(self.dropout_vec)])

                    with tf.name_scope("Network"):
                        self._network_tensor = load_imitation_learning_network(model_inputs_mode, self._input_images, 
                                                                               self._input_data[1],
                                                                               self._dout, is_training=False)

                    self._models_path = model_path

                    # tf.reset_default_graph()
                    sess.run(tf.global_variables_initializer())

                    self.load_model()

    def load_model(self):

        variables_to_restore = tf.global_variables()

        saver = tf.train.Saver(variables_to_restore, max_to_keep=0)

        if not os.path.exists(self._models_path):
            raise RuntimeError('failed to find the models path')

        ckpt = tf.train.get_checkpoint_state(self._models_path)
        if ckpt:
            print('Restoring from ', ckpt.model_checkpoint_path)
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            ckpt = 0

        return ckpt

    def run_step(self, measurements, sensor_data, directions, target):

        control, pred_speed = self._compute_action(sensor_data['CameraRGB'].data,
                                              measurements.player_measurements.forward_speed, directions)

        return control, pred_speed

    def _compute_action(self, rgb_image, actual_speed, direction=None):

        rgb_image = rgb_image[self._image_cut[0]:self._image_cut[1], :]

        image_input = scipy.misc.imresize(rgb_image, [self._image_size[0], self._image_size[1]])

        visualize = False
        if visualize:
            import matplotlib.pyplot as plt
            plt.imshow(image_input)
            plt.show(block=False)

        image_input = image_input.astype(np.float32)
        image_input = np.multiply(image_input, 1.0 / 255.0)

        steer, acc, brake, pred_speed = self._control_function(image_input, actual_speed, direction, self.sess)

        # This a bit biased, but is to avoid fake breaking

        if brake < 0.1:
            brake = 0.0

        if acc > brake:
            brake = 0.0

        # We limit actual_speed to 35 km/h to avoid
        if actual_speed > 10.0 and brake == 0.0:
            acc = 0.0

        control = Control()
        control.steer = steer
        control.throttle = acc
        control.brake = brake

        control.hand_brake = 0
        control.reverse = 0

        return control, pred_speed

    def _control_function(self, image_input, actual_speed, control_input, sess):

        branches = self._network_tensor
        x = self._input_images
        dout = self._dout
        input_speed = self._input_data[1]

        image_input = image_input.reshape(
            (1, self._image_size[0], self._image_size[1], self._image_size[2]))

        # actual_speed variable is actual car speed in m/s
        # Normalize with the maximum actual_speed from the training set ( 90 km/h = 25m/s)
        actual_speed = np.array(actual_speed / 25.0)
        actual_speed = actual_speed.reshape((1, 1))

        # control_input is:
        # LANE_FOLLOW = 2.0
        # REACH_GOAL = 0.0
        # TURN_LEFT = 3.0
        # TURN_RIGHT = 4.0
        # GO_STRAIGHT = 5.0
        if self._load_ready_model:
            if control_input == 2 or control_input == 0.0:
                all_net = branches[0]
            elif control_input == 3:
                all_net = branches[2]
            elif control_input == 4:
                all_net = branches[3]
            else:
                all_net = branches[1]
        else:
            # control_input should be mapped to this branch configuration:
            # ['Go Right', 'Go Straight', 'Follow Lane', 'Go Left', 'Speed Prediction Branch']
            if control_input == 2 or control_input == 0.0:
                all_net = branches[2]
            elif control_input == 3:
                all_net = branches[3]
            elif control_input == 4:
                all_net = branches[0]
            else:
                all_net = branches[1]
                # all_net = branches[2]  # Comment this, it's for debugging

        feedDict = {x: image_input, input_speed: actual_speed, dout: [1] * len(self.dropout_vec)}

        output_all = sess.run(all_net, feed_dict=feedDict)

        # Visualize for debugging
        visualize = False
        if visualize:
            import matplotlib.pyplot as plt
            plt.cla()
            plt.imshow(image_input[0])
            print(output_all[0])
            plt.pause(0.1)
            plt.draw()

        predicted_steers = (output_all[0][0])
        predicted_acc = (output_all[0][1])
        # predicted_acc = 0.25  # Comment this, it's for debugging
        predicted_brake = (output_all[0][2])

        predicted_speed = sess.run(branches[4], feed_dict=feedDict)
        predicted_speed = predicted_speed[0][0]
        if self._avoid_stopping:
            real_speed = actual_speed * 25.0  # Unnormalize: now it is m/s, not from 0 to 1
            real_predicted = predicted_speed * 25.0  # Unnormalize: now it is m/s, not from 0 to 1
            # TODO: why 0.4 m/s gives betetr results than 3.0 m/s?
            # If (Car Stooped) and ( It should not have stopped because network predicts this):
            if real_speed < 2.0 and \
                    ((self._load_ready_model and real_predicted > 3.0) or
                     ((not self._load_ready_model) and real_predicted > 0.4)):  # 3.0 or 0.4
                # Increase acceleration by: 1 * (5.6 m/s - car actual speed m/s)
                predicted_acc = 1 * (5.6 / 25.0 - actual_speed) + predicted_acc

                predicted_brake = 0.0  # Force no brake

                predicted_acc = predicted_acc[0][0]

        return predicted_steers, predicted_acc, predicted_brake, predicted_speed
