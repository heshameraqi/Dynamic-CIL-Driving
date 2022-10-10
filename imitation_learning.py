from __future__ import print_function

import os
import matplotlib.pyplot as plt
import scipy
import tensorflow as tf
import numpy as np

from carla.agent import Agent
from carla.carla_server_pb2 import Control
from network import load_imitation_learning_network
from carla import image_converter


class ImitationLearning(Agent):
    '''load_ready_model: because ready model assumed a different order of network output branches'''

    def __init__(self, model_inputs_mode, our_experiment_sensor_setup, model_path, avoid_stopping, memory_fraction=0.25,
                 image_cut=[115, 510], load_ready_model=False, gpu=0, gpu_memory_fraction=0.4):

        Agent.__init__(self)

        self.model_inputs_mode = model_inputs_mode
        self.our_experiment_sensor_setup = our_experiment_sensor_setup
        self._image_cut = image_cut
        # Just the length of the following dropout_vec is used
        if self.model_inputs_mode == "1cam":
            self.dropout_vec = [1.0] * 8 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.0] * 5  # keep probabilities
        elif self.model_inputs_mode == "1cam-pgm":
            self.dropout_vec = [1.0] * 8 + [1.0] * 4 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5,
                                                                                            1.0] * 5  # keep probabilities
        elif self.model_inputs_mode == "3cams":
            self.dropout_vec = [1.0] * 8 * 3 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.0] * 5  # keep probabilities
        elif self.model_inputs_mode == "3cams-pgm":
            self.dropout_vec = [1.0] * 8 * 3 + [1.0] * 4 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5,
                                                                                                1.0] * 5  # keep probabilities
        else:
            assert ("Bad value set for model_inputs_mode! Exiting ...")
            exit()

        # Seen GPU's and memory_fraction used are selected
        # tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.visible_device_list = str(gpu)
        tf_config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
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
                    self._input_images_right = tf.placeholder("float", shape=[None, self._image_size[0],
                                                                              self._image_size[1],
                                                                              self._image_size[2]],
                                                              name="input_image_right")
                    self._input_images_left = tf.placeholder("float", shape=[None, self._image_size[0],
                                                                             self._image_size[1],
                                                                             self._image_size[2]],
                                                             name="input_image_left")
                    self._input_images_lidar_pgm = tf.placeholder("float", shape=[None, 32, 90],
                                                                  # TODO: make it not hard coded
                                                                  name="input_image_lidar_pgm")

                    self._input_data = []
                    self._input_data.append(tf.placeholder(tf.float32,
                                                           shape=[None, 4], name="input_control"))

                    self._input_data.append(tf.placeholder(tf.float32,
                                                           shape=[None, 1], name="input_speed"))

                    self._dout = tf.placeholder("float", shape=[len(self.dropout_vec)])

                    with tf.name_scope("Network"):
                        self._network_tensor = load_imitation_learning_network(self.model_inputs_mode,
                                                                               self._input_images,
                                                                               self._input_data[1],
                                                                               self._dout, is_training=False,
                                                                               input_image_right=self._input_images_right,
                                                                               input_image_left=self._input_images_left,
                                                                               input_image_lidar_pgm=self._input_images_lidar_pgm)

                    self._models_path = model_path

                    # tf.reset_default_graph()
                    sess.run(tf.global_variables_initializer())

                    self.load_model()

    def load_model(self):

        variables_to_restore = tf.global_variables()

        saver = tf.train.Saver(variables_to_restore, max_to_keep=0)

        if not os.path.exists(self._models_path):
            raise RuntimeError('failed to find the models path: ' + self._models_path)

        ckpt = tf.train.get_checkpoint_state(self._models_path)
        if ckpt:
            print('Restoring from ', self._models_path + '/' + os.path.basename(ckpt.model_checkpoint_path))
            saver.restore(self.sess, self._models_path + '/' + os.path.basename(ckpt.model_checkpoint_path))
        else:
            ckpt = 0

        return ckpt

    @staticmethod
    # Needed for LiDAR PGM
    def cart2pol(x, y, z):
        xy = x ** 2 + y ** 2
        rho = np.sqrt(xy + z ** 2)
        theta = np.arctan2(y, x)
        phi = np.arctan2(z, np.sqrt(xy))  # np.arctan2 retruns from -np.pi to np.pi

        # make angles from 0 to 360
        theta_deg = (np.degrees(theta) + 360) % 360
        phi_deg = (np.degrees(phi) + 360) % 360

        return rho, theta_deg, phi_deg

    def run_step(self, measurements, sensor_data, directions, target):
        _image_cam_centre = image_converter.to_rgb_array(sensor_data.get('CameraRGB_centre', None))
        # _image_cam_depth = image_converter.depth_to_logarithmic_grayscale(sensor_data.get('CameraDepth', None))
        # _image_cam_semsegm = image_converter.labels_to_cityscapes_palette(sensor_data.get('CameraSemSeg', None))
        _image_cam_right = None
        _image_cam_left = None
        if self.model_inputs_mode == "3cams" or self.model_inputs_mode == "3cams-pgm":
            _image_cam_right = image_converter.to_rgb_array(sensor_data.get('CameraRGB_right', None))
            _image_cam_left = image_converter.to_rgb_array(sensor_data.get('CameraRGB_left', None))

        _lidar_pgm_image = None
        if self.model_inputs_mode == "3cams-pgm" or self.model_inputs_mode == "1cam-pgm":
            # LiDAR PGM generation
            # Configurations
            # should be the same as during data collection automatically (_collect_data.py)
            horizontal_angle_step = 2  # in degrees, should be divisible by 180
            nbrLayers = 32
            upper_fov = 10  # in degrees
            lower_fov = -30  # in degrees
            # lidar_range = 50
            # unreflected_value = float(lidar_range + 10)  # PGM value when no beam received
            unreflected_value = 50

            _lidar_measurement = sensor_data.get('Lidar', None)
            self.lidar_data = np.array(_lidar_measurement.data)
            x = self.lidar_data[:, 0]
            y = self.lidar_data[:, 1]
            z = -self.lidar_data[:, 2]
            (rho, theta_deg, phi_deg) = ImitationLearning.cart2pol(x, y, z)  # theta of 90 means front-facing
            unique_phis = (np.arange(upper_fov, lower_fov, -(upper_fov - lower_fov) / nbrLayers) -
                           (upper_fov - lower_fov) / (2. * nbrLayers) + 360) % 360
            unique_thetas = np.arange(180, 360,
                                      horizontal_angle_step) + horizontal_angle_step / 2.  # only range front-facing
            _lidar_pgm_image = np.ones((len(unique_phis), len(unique_thetas))) * unreflected_value
            for i in range(_lidar_pgm_image.shape[0]):  # For each layer
                for j in range(_lidar_pgm_image.shape[1]):  # For each group of neighboring beams
                    # TODO:I found that multiplying with 1.1 (a number slightly bigger than 1) prevents PGM artifacts, why?
                    indices_phi = np.abs(phi_deg - unique_phis[i]) <= 1.1 * (upper_fov - lower_fov) / (2. * nbrLayers)
                    indices_theta = np.abs(theta_deg - unique_thetas[j]) <= horizontal_angle_step / 2.
                    rhos = rho[indices_phi & indices_theta]
                    if len(rhos) > 0:
                        _lidar_pgm_image[i, j] = np.mean(rhos)
            _lidar_pgm_image = 255.0 * _lidar_pgm_image / unreflected_value
            # _lidar_pgm_image = 255.0 * np.repeat(_lidar_pgm_image[:, :, np.newaxis], 3, axis=2) / unreflected_value

        control, pred_speed = self._compute_action(_image_cam_centre, _image_cam_right, _image_cam_left,
                                                   _lidar_pgm_image, measurements, directions, sensor_data)

        return control, pred_speed, _lidar_pgm_image

    def _compute_action(self, image_input, image_input_right, image_input_left, image_input_lidar_pgm,
                        measurements, direction, sensor_data):
        if not self.our_experiment_sensor_setup:
            image_input = image_input[self._image_cut[0]:self._image_cut[1], :]

        image_input = scipy.misc.imresize(image_input, [self._image_size[0], self._image_size[1]])
        if self.model_inputs_mode == "3cams" or self.model_inputs_mode == "3cams-pgm":
            image_input_right = scipy.misc.imresize(image_input_right, [self._image_size[0], self._image_size[1]])
            image_input_left = scipy.misc.imresize(image_input_left, [self._image_size[0], self._image_size[1]])
        # if self.model_inputs_mode == "3cams-pgm" or self.model_inputs_mode == "1cam-pgm":
        #     image_input_lidar_pgm = scipy.misc.imresize(image_input_lidar_pgm, [self._image_size[0], self._image_size[1]])

        # For debugging, save sample images (Comment this)
        '''from scipy import misc
        misc.imsave('/home/heraqi/temp/00000DEPLOY.png', image_input)
        exit()'''

        visualize = False
        if visualize:
            plt.figure()
            plt.imshow(image_input)
            plt.show(block=False)
            plt.figure()
            plt.imshow(image_input_right)
            plt.show(block=False)
            plt.figure()
            plt.imshow(image_input_left)
            plt.show(block=False)
            print(np.min(image_input))
            print(np.min(image_input_right))
            print(np.min(image_input_left))
            print(np.max(image_input))
            print(np.max(image_input_right))
            print(np.max(image_input_left))
            import pdb
            pdb.set_trace()

        image_input = image_input.astype(np.float32)
        image_input = np.multiply(image_input, 1.0 / 255.0)
        if self.model_inputs_mode == "3cams" or self.model_inputs_mode == "3cams-pgm":
            image_input_right = image_input_right.astype(np.float32)
            image_input_right = np.multiply(image_input_right, 1.0 / 255.0)
            image_input_left = image_input_left.astype(np.float32)
            image_input_left = np.multiply(image_input_left, 1.0 / 255.0)
        if self.model_inputs_mode == "3cams-pgm" or self.model_inputs_mode == "1cam-pgm":
            image_input_lidar_pgm = image_input_lidar_pgm.astype(np.float32)
            image_input_lidar_pgm = np.multiply(image_input_lidar_pgm, 1.0 / 255.0)

        steer, acc, brake, pred_speed = self._control_function(image_input, image_input_right, image_input_left,
                                                               image_input_lidar_pgm, measurements, direction,
                                                               self.sess, sensor_data)

        # This a bit biased, but is to avoid fake breaking
        if brake < 0.1:
            brake = 0.0
        if acc > brake:
            brake = 0.0

        # We limit actual_speed to 35 km/h to avoid
        if measurements.player_measurements.forward_speed > 10.0 and brake == 0.0:  # forward_speed is actual_speed
            acc = 0.0

        control = Control()
        control.steer = steer
        control.throttle = acc
        control.brake = brake

        control.hand_brake = 0
        control.reverse = 0

        return control, pred_speed

    def decode_segmap(self, image, nc=21):
        label_colors = np.array([(0, 0, 0),  # 0=background
                                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                                 (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                                 (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                                 # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                                 (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                                 # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                                 (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

        r = np.zeros_like(image).astype(np.uint8)
        g = np.zeros_like(image).astype(np.uint8)
        b = np.zeros_like(image).astype(np.uint8)

        for l in range(0, nc):
            idx = image == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]

        rgb = np.stack([r, g, b], axis=2)
        return rgb

    def _control_function(self, image_input, image_input_right, image_input_left, image_input_lidar_pgm, measurements,
                          control_input, sess, sensor_data):

        actual_speed = measurements.player_measurements.forward_speed

        branches = self._network_tensor

        x = self._input_images
        x_right_image = self._input_images_right
        x_left_image = self._input_images_left
        x_lidar_pgm_image = self._input_images_lidar_pgm

        dout = self._dout
        input_speed = self._input_data[1]

        image_input = image_input.reshape((1, self._image_size[0], self._image_size[1], self._image_size[2]))
        if self.model_inputs_mode == "3cams" or self.model_inputs_mode == "3cams-pgm":
            image_input_right = image_input_right.reshape(
                (1, self._image_size[0], self._image_size[1], self._image_size[2]))
            image_input_left = image_input_left.reshape(
                (1, self._image_size[0], self._image_size[1], self._image_size[2]))
        if self.model_inputs_mode == "3cams-pgm" or self.model_inputs_mode == "1cam-pgm":
            image_input_lidar_pgm = image_input_lidar_pgm.reshape((1, image_input_lidar_pgm.shape[0],
                                                                   image_input_lidar_pgm.shape[1]))

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

        if self.model_inputs_mode == "1cam":
            feedDict = {x: image_input, input_speed: actual_speed, dout: [1] * len(self.dropout_vec)}
        elif self.model_inputs_mode == "1cam-pgm":
            feedDict = {x: image_input, x_lidar_pgm_image: image_input_lidar_pgm, input_speed: actual_speed,
                        dout: [1] * len(self.dropout_vec)}
        elif self.model_inputs_mode == "3cams":
            feedDict = {x: image_input, x_right_image: image_input_right, x_left_image: image_input_left,
                        input_speed: actual_speed, dout: [1] * len(self.dropout_vec)}
        elif self.model_inputs_mode == "3cams-pgm":
            feedDict = {x: image_input, x_right_image: image_input_right, x_left_image: image_input_left,
                        x_lidar_pgm_image: image_input_lidar_pgm, input_speed: actual_speed,
                        dout: [1] * len(self.dropout_vec)}

        output_all = sess.run(all_net, feed_dict=feedDict)

        # Visualize for debugging
        visualize = False
        if visualize:
            img = image_input[0]
            plt.cla()
            plt.imshow(img)
            print('min=' + str(np.min(img)) + ', max=' + str(np.max(img)))
            print("predictions: ")
            print(output_all[0])
            plt.pause(0.05)
            plt.draw()
            plt.show()
            # plt.waitforbuttonpress()

        predicted_steers = (output_all[0][0])
        predicted_acc = (output_all[0][1])
        # predicted_acc = 0.25  # Comment this, it's for debugging
        predicted_brake = (output_all[0][2])

        predicted_speed = sess.run(branches[4], feed_dict=feedDict)
        predicted_speed = predicted_speed[0][0]

        if self._avoid_stopping:
            real_speed = actual_speed * 25.0  # Unnormalize: now it is m/s, not from 0 to 1
            real_predicted = predicted_speed * 25.0  # Unnormalize: now it is m/s, not from 0 to 1

            if real_speed < 2.0:  # If car stops
                # TODO: why 0.4 m/s gives better results than 3.0 m/s?
                # 3.0 (high threshold, default) or 0.4 (low threshold)
                # (self._load_ready_model and real_predicted > 3.0) or ((not self._load_ready_model) and real_predicted > 0.4):
                if real_predicted > 3.0:
                    # Increase acceleration to reach speed of 5.6 m/s (if multiply factor = 1)
                    predicted_acc_old = predicted_acc
                    predicted_acc = 1 * (5.6 / 25.0 - actual_speed[0][0]) + predicted_acc
                    # print("Prevented stopping due to high predicted speed value! Acc=%f-->%f" %
                    #       (predicted_acc_old, predicted_acc))
                    predicted_brake = 0.0  # Force no brake

            # TODO: why 0.4 m/s gives betetr results than 3.0 m/s?
            # If (Car Stooped) and ( It should not have stopped because network predicts this):

        # Don't drive fast
        # if real_speed >= 10 or predicted_acc>=0.8:
        #     predicted_acc /= 2.0

        return predicted_steers, predicted_acc, predicted_brake, predicted_speed
