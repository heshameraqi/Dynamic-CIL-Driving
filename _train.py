# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------
import os
import sys
import glob
import hashlib
import time
import numpy as np
from Data import Data
from network import load_imitation_learning_network
import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
import pickle
import matplotlib.pyplot as plt
import random
import shutil


# ------------------------------------------------------------------------------
# Configurations
# ------------------------------------------------------------------------------
class Training_configs:
    def __init__(self):
        # Part 1/2: Configurations that will affect the experiment hash
        self.train_data = "auc2"  # "auc1" (collect_data1.py) or "auc2" (collect_data2.py) or "il_dataset"
        if self.train_data == "auc1":
            self.datasetDirTrain = '/media/heraqi/data/heraqi/int-end-to-end-ad/auc.carla.dataset_00/train'
            self.datasetDirVal = '/media/heraqi/data/heraqi/int-end-to-end-ad/auc.carla.dataset_00/val'
            self.meta_data_filename = '/metadata_with_highlvlcmd.csv'
            self.img_width = 200  # 320
            self.img_height = 88  # 240 (qVGA)
        elif self.train_data == "auc2":  # LiDAR PGM is 90 x 32
            self.datasetDirTrain = '/media/heraqi/data/heraqi/int-end-to-end-ad/auc.carla.dataset_01/train'
            self.datasetDirVal = '/media/heraqi/data/heraqi/int-end-to-end-ad/auc.carla.dataset_01/val'
            self.meta_data_filename = 'metadata.json'
            self.img_width = 200  # 320
            self.img_height = 88  # 240 (qVGA)
        elif self.train_data == "il_dataset":
            self.datasetDirTrain = '/media/heraqi/data/heraqi/AgentHuman/SeqTrain'
            self.datasetDirVal = '/media/heraqi/data/heraqi/AgentHuman/SeqVal'
            self.meta_data_filename = ''
            self.img_width = 200  # 320
            self.img_height = 88  # 240 (qVGA)
        self.augmentation_ratio = 0.5  # 0.5
        self.model_inputs_mode = "1cam-pgm"  # "1cam" or "1cam-pgm" or "3cams" or "3cams-pgm"
        self.cBranchesOutList = ['Go Right', 'Go Straight', 'Follow Lane', 'Go Left', 'Speed Prediction Branch']
        self.batch_size = 120  # should be multiple of 4 (commands) or multiple of nbr_steering_bins if bins by steers
        self.augment_data = True
        self.single_branch_per_minibatch = False
        self.nbr_steering_bins = 8
        self.enable_batch_norm = True
        self.learning_rate = 0.0002
        self.divide_learning_rate_after_epochs = 15
        self.divide_learning_rate_by = 1.5
        self.learning_rate_min = 0.00002
        if self.model_inputs_mode == "1cam":
            self.dropout_vec = [1.0] * 8                 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.0] * 5  # keep probabilities
        elif self.model_inputs_mode == "1cam-pgm":
            # self.dropout_vec = [1.0] * 8 + [1.0] * 8     + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.0] * 5  # TEST 1
            self.dropout_vec = [1.0] * 8 + [1.0] * 4     + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.0] * 5  # keep probabilities
        elif self.model_inputs_mode == "3cams":
            self.dropout_vec = [1.0] * 8 * 3             + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.0] * 5  # keep probabilities
        elif self.model_inputs_mode == "3cams-pgm":
            self.dropout_vec = [1.0] * 8 * 3 + [1.0] * 4 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.0] * 5  # keep probabilities
        else:
            assert ("Bad value set for model_inputs_mode! Exiting ...")
            exit()
        self.beta1 = 0.7  # TODO Tune it
        self.beta2 = 0.85  # TODO Tune it
        self.branchConfigs = [["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"],
                              ["Steer", "Gas", "Brake"], ["Steer", "Gas", "Brake"], ["Speed"]]

        self.optimizer = 'ADAM'  # TODO: Not used
        self.beta1 = 0.7
        self.beta2 = 0.85
        self.loss = 'L2'  # TODO: Not used
        self.L2NormConst = 0.001

        # Get experiment hash to store results, and save config file
        # self.modelsPath = '/media/heraqi/data/heraqi/int-end-to-end-ad/models'  # Without / at the end
        self.modelsPath = "/home/heraqi/scripts/models"  # Without / at the end
        self.gen_mdl_hash()  # get training/experiment configurations hash

        # Part 2/2: Configurations that won't affect the experiment hash
        self.gpu_id = 0  # GPU for running the the model, from 0 to 7 in DGX-1 server
        self.gpu_memory_fraction = 0.25  # 0.25 for 1 CNN stream model, 0.4 for the bigger
        self.no_epochs = 100
        self.load_model = False
        self.model_to_load = '/media/heraqi/data/heraqi/int-end-to-end-ad/models/F3F1E4_AUC2_data_1cam_27/epoch_31'
        self.save_each_epoch = 1  # Save each epoch number save_each_epoch
        self.memory_fraction = 1.0
        self.logsPath = './logs'
        # self.log_level = 'debug'  # 'debug' or 'info', debug is more detailed
        # self.host = '127.0.0.1'
        # self.port = 2000

    # Get hash, and save configurations and configurations object
    def gen_mdl_hash(self):
        hash_md5 = hashlib.md5()

        parameters_text = ""

        members = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        for m in members:
            if m.startswith("modelsPath"):
                continue
            if isinstance(m, int):
                hash_md5.update(("%i" % m).encode())
                parameters_text += "%s = %i\n" % (m, getattr(self, m))
            elif isinstance(m, bool):
                if m:
                    hash_md5.update("True".encode())
                    parameters_text += "%s = True\n" % m
                else:
                    hash_md5.update("False".encode())
                    parameters_text += "%s = False\n" % m
            elif isinstance(m, list):  # numeric 1D list is assumed
                for i in range(len(m)):
                    hash_md5.update(("%i" % i).encode())
                    parameters_text += "%s[%i] = %i\n" % (m, i, getattr(self, m[i]))
            else:  # should be string
                hash_md5.update(m.encode())
                parameters_text += "%s = %s\n" % (m, getattr(self, m))

        hex_dig = hash_md5.hexdigest()
        self.experiment_MDL_hash = ("%s" % hex_dig[0:6]).upper()
        if self.train_data == "auc1":
            self.experiment_MDL_hash += '_AUC_data_' + self.model_inputs_mode
        elif self.train_data == "auc2":
            self.experiment_MDL_hash += '_AUC2_data_' + self.model_inputs_mode
        elif self.train_data == "il_dataset":
            self.experiment_MDL_hash += '_CIL_data'

        # If a directory exists with same hash add _i to hash
        tmpDir = self.modelsPath + "/" + self.experiment_MDL_hash
        cnt = 0
        while os.path.isdir(tmpDir):
            cnt += 1
            tmpDir = self.modelsPath + "/" + self.experiment_MDL_hash + "_%i" % cnt
        if cnt != 0:
            self.experiment_MDL_hash += "_%i" % cnt
        '''if os.path.exists(self.modelsPath + "/" + self.experiment_MDL_hash):
            shutil.rmtree(self.modelsPath + "/" + self.experiment_MDL_hash, ignore_errors=True)'''

        print("Saving experiment parameters file to: " + self.modelsPath + "/" + self.experiment_MDL_hash)
        os.makedirs(self.modelsPath + "/" + self.experiment_MDL_hash)
        with open(self.modelsPath + "/" + self.experiment_MDL_hash + "/configs.txt", 'w') as output_file:
            output_file.write(parameters_text)
        with open(self.modelsPath + "/" + self.experiment_MDL_hash + "/configs_object.pkl", "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


# ------------------------------------------------------------------------------
# Create Network function
# ------------------------------------------------------------------------------
def create_network(training_configs, is_training=True):
    # Create inputs
    input_images = tf.placeholder("float", shape=[None, training_configs.img_height, training_configs.img_width, 3],
                                 name="input_image")
    input_images_right = tf.placeholder("float",
                                        shape=[None, training_configs.img_height, training_configs.img_width, 3],
                                        name="input_image_right")
    input_images_left = tf.placeholder("float",
                                       shape=[None, training_configs.img_height, training_configs.img_width, 3],
                                       name="input_image_left")
    input_images_pgm = tf.placeholder("float",
                                      shape=[None, 32, 90],  # TODO: this needs to be not hard coded
                                      name="input_image_lidar_pgm")
    input_measurements = tf.placeholder(tf.float32, shape=[None, 1], name="input_speed")
    inputs = [input_images, input_measurements, input_images_right, input_images_left, input_images_pgm]

    # Branch Controller
    high_lvl_cmds = tf.placeholder(tf.int32, shape=[None], name="input_control")

    # Create targets
    targetController = tf.placeholder(tf.float32, shape=[None, 3], name="target_control")
    targetSpeed = tf.placeholder(tf.float32, shape=[None, 1], name="target_speed")
    targets = [targetController, targetSpeed]
    targets = [targetController, targetSpeed]

    # Create dropout configuration tensor and learning rate place holder
    drop_outs = tf.placeholder(tf.float32, shape=[len(training_configs.dropout_vec)])
    LR = tf.placeholder(tf.float32, shape=[])

    # Create network on top of imitation_learning_network
    with tf.name_scope("Network"):
        network_branches = load_imitation_learning_network(training_configs.model_inputs_mode, inputs[0], inputs[1],
                                                           dropout_vector=drop_outs,
                                                           branch_config=training_configs.branchConfigs,
                                                           is_training=is_training, input_image_right=inputs[2],
                                                           input_image_left=inputs[3], input_image_lidar_pgm=inputs[4])
        # Branching and loss calculation
        branches_losses = []
        speed_part = None
        for i in range(0, len(training_configs.branchConfigs)):
            with tf.name_scope("Branch_" + str(i)):
                print(training_configs.branchConfigs[i])
                if training_configs.branchConfigs[i][0] == "Speed":
                    speed_part = tf.square(tf.subtract(network_branches[-1], targets[1]))
                else:
                    branch_loss = tf.square(tf.subtract(network_branches[i], targets[0]))
                    branches_losses.append(branch_loss)
        branches_losses_tensor = tf.convert_to_tensor(branches_losses)
        high_lvl_cmds_mask = tf.one_hot(high_lvl_cmds, 4)
        high_lvl_cmds_mask = tf.convert_to_tensor(high_lvl_cmds_mask)
        pr = tf.print(high_lvl_cmds_mask, [high_lvl_cmds_mask], summarize=5)  # one hot vector of branch number

        # Averaging loss (old method, which I believe not correct,
        # GitHub issue: https://github.com/mvpcom/carlaILTrainer/issues/19#issuecomment-504052075
        # loss = tf.reduce_mean(tf.multiply(tf.transpose(tf.reduce_mean(branches_losses_tensor, 2)), high_lvl_cmds_mask))
        # if speed_part is not None:
        #     loss = tf.reduce_mean([loss, tf.reduce_mean(speed_part)])

        # Calculate losses: loss_steer is the loss for each image in the minibatch,
        # represented by the squared error from ground-truth
        loss_steer = tf.reduce_sum(tf.multiply(tf.transpose(branches_losses_tensor[:, :, 0]), high_lvl_cmds_mask),
                                   axis=1)
        loss_throttle = tf.reduce_sum(tf.multiply(tf.transpose(branches_losses_tensor[:, :, 1]), high_lvl_cmds_mask),
                                      axis=1)
        loss_brake = tf.reduce_sum(tf.multiply(tf.transpose(branches_losses_tensor[:, :, 2]), high_lvl_cmds_mask),
                                   axis=1)
        loss_speed = tf.reduce_sum(speed_part, axis=1)

        # Weighted averaging the loss
        batch_size = tf.shape(loss_steer)[0]
        weights = tf.reshape(tf.tile(tf.constant([0.5, 0.2, 0.15, 0.15]), [batch_size]), [batch_size, 4])
        loss = tf.multiply(tf.stack([loss_steer, loss_throttle, loss_brake, loss_speed]), tf.transpose(weights))
        loss = tf.reduce_sum(loss, axis=0)
        loss = tf.reduce_mean(loss)

        '''loss = 0.5 * tf.reduce_mean(loss_steer) + 0.2 * tf.reduce_mean(loss_throttle) + \
                  0.15 * tf.reduce_mean(loss_brake) + 0.15 * tf.reduce_mean(loss_speed)'''
        '''losses = tf.stack([tf.reduce_mean(loss_steer), tf.reduce_mean(loss_throttle), tf.reduce_mean(loss_brake),
                           tf.reduce_mean(loss_speed)])
        loss = 0.5*losses[0] + 0.2*losses[1] + 0.15*losses[2] + 0.15*losses[3]'''

        # For debugging, should be commented
        # internal_losses = {'loss_steer': loss_steer, 'loss_throttle': loss_throttle, 'loss_brake': loss_brake,
        #                    'loss_speed': loss_speed, 'losses': losses, 'loss': loss}

        # Optimizer
        contSolver = tf.train.AdamOptimizer(learning_rate=LR,
                                            beta1=training_configs.beta1,
                                            beta2=training_configs.beta2).minimize(loss)

    tensors = {
        'inputs': inputs,
        'high_lvl_cmds': high_lvl_cmds,
        'targets': targets,
        'drop_outs': drop_outs,
        'LR': LR,
        'optimizers': contSolver,
        'losses': loss,
        # 'losses': internal_losses,  # For debugging, should be commented
        'network_branches': network_branches,
        'print': pr
    }
    return tensors


# ------------------------------------------------------------------------------
# Validate model function
# ------------------------------------------------------------------------------
def validate_model(iterate_batches_train, iterate_batches_val):
    print("Validating the model ...")
    train_data_sample_percent = 30  # 30 Random sample of training data to validate against (to save time)
    batch_size = 1000  # 1000 (Titan-X, with the GPU fraction used) or 5000 (DGX-1)

    train_losses = []
    mb = 1
    for train_data, nbrBatches_train in iterate_batches_train(shuffle_data=True, augment_data=False,
                                                              batch_size=batch_size,
                                                              sample_percent=train_data_sample_percent):
        images_cam_1, images_cam_2, images_cam_3, images_pgm, steers, throttles, brakes, speeds, \
        high_lvl_cmds, _ = train_data
        feedDict = {netTensors['inputs'][0]: images_cam_1,
                    netTensors['inputs'][1]: speeds.reshape((-1, 1)),
                    netTensors['inputs'][2]: images_cam_2,
                    netTensors['inputs'][3]: images_cam_3,
                    netTensors['inputs'][4]: images_pgm,
                    netTensors['high_lvl_cmds']: high_lvl_cmds,
                    # netTensors['high_lvl_cmds']: np.zeros(training_configs.batch_size),  # To train a single branch for now!
                    netTensors['drop_outs']: [1] * len(training_configs.dropout_vec),
                    netTensors['targets'][0]: np.hstack([steers.reshape((-1, 1)),
                                                         throttles.reshape((-1, 1)),
                                                         brakes.reshape((-1, 1))]),
                    netTensors['targets'][1]: speeds.reshape((-1, 1))}
        loss = sess.run(contLoss, feed_dict=feedDict)
        train_losses.append(loss)
        print("    Validating Model on Training Data::: Epoch: %d & Minibatch: %d/%d (Step: %d), "
              "Portion: %d/%d, Partial Loss: %g" % (epoch + 1, mini_batch, num_of_batches, steps, mb,
                                                    nbrBatches_train, loss))
        mb += 1
    train_loss = np.mean(train_losses)

    val_losses = []
    mb = 1
    for val_data, nbrBatches_val in iterate_batches_val(shuffle_data=True, augment_data=False,
                                                        batch_size=batch_size,
                                                        sample_percent=100):
        images_cam_1, images_cam_2, images_cam_3, images_pgm, steers, throttles, brakes, speeds, \
        high_lvl_cmds, _ = val_data
        feedDict = {netTensors['inputs'][0]: images_cam_1,
                    netTensors['inputs'][1]: speeds.reshape((-1, 1)),
                    netTensors['inputs'][2]: images_cam_2,
                    netTensors['inputs'][3]: images_cam_3,
                    netTensors['inputs'][4]: images_pgm,
                    netTensors['high_lvl_cmds']: high_lvl_cmds,
                    netTensors['drop_outs']: [1] * len(training_configs.dropout_vec),
                    netTensors['targets'][0]: np.hstack([steers.reshape((-1, 1)),
                                                         throttles.reshape((-1, 1)),
                                                         brakes.reshape((-1, 1))]),
                    netTensors['targets'][1]: speeds.reshape((-1, 1))}
        loss = sess.run(contLoss, feed_dict=feedDict)
        val_losses.append(loss)
        print("    Validating Model on Validation Data::: Epoch: %d & Minibatch: %d/%d (Steps: %d),"
              " Portion: %d/%d, Partial Loss: %g" % (epoch + 1, mini_batch, num_of_batches, steps,
                                                     mb, nbrBatches_val, loss))
        mb += 1
    val_loss = np.mean(val_losses)

    print("Validation finished: train data loss=%g, validation data loss=%g" % (train_loss, val_loss))
    return train_loss, val_loss


def visualize_data(data, train_data, display_shuffled_samples=False, delay=2):
    if train_data == 'auc2':
        store_folder = '/media/heraqi/data/heraqi/int-end-to-end-ad/vis_batches_auc2_data'
    elif 'il_dataset':
        store_folder = '/media/heraqi/data/heraqi/int-end-to-end-ad/vis_batches_cil_data'
    if os.path.exists(store_folder):
        shutil.rmtree(store_folder, ignore_errors=True)
    os.mkdir(store_folder)

    images_cam_1, images_cam_2, images_cam_3, images_pgm, steers, throttles, brakes, speeds, \
    high_lvl_cmds, data_indices = data

    print("image max value=" + str(np.max(images_cam_1)))

    fig, ax = plt.subplots(2, 3, figsize=(12, 6))
    ax[0, 0].set_title('Left camera')
    ax[0, 1].set_title('Centre camera')
    ax[0, 2].set_title('Right camera')
    ax[1, 1].set_title('PGM LiDAR view')
    r = list(range(len(images_cam_1)))
    if display_shuffled_samples:
        random.shuffle(r)
    for i in r:
        # Target specific case to visualize
        if True:
        # if  brakes[i] == 1:
        # if np.abs(steers[i]) > 0.3 and high_lvl_cmds[i] == 0:  # cmd==3:Left - cmd==0:Right
            ax[0, 0].imshow(images_cam_3[i])  # images_cam_3: Left
            ax[0, 0].set_aspect('equal')
            ax[0, 1].imshow(images_cam_1[i])  # images_cam_1: Centre
            ax[0, 1].set_aspect('equal')
            ax[0, 2].imshow(images_cam_2[i])  # images_cam_2: Right
            ax[0, 2].set_aspect('equal')
            ax[1, 1].imshow(images_pgm[i])  # PGM image
            ax[1, 1].set_aspect('equal')
            if train_data == 'auc1':
                folder_index = int(data_indices[
                                       i] / 4500)  # TODO: that hardcoded number should be the size of each episode for the bigger data in AUC server
                file_nbr = (data_indices[i] % 4500) + 1
                fig.suptitle("Folder index=%d, File number=%d, High_lvl_cmd=%.12s, Steer=%03.2f, Throttle=%03.2f, "
                             "Brake=%02.1f, Speed=%04.2f" %
                             (folder_index, file_nbr, training_configs.cBranchesOutList[int(high_lvl_cmds[i])],
                              steers[i],
                              throttles[i], brakes[i], speeds[i]))
            elif train_data == 'auc2':
                fig.suptitle("Data sample index=%s\nHigh_lvl_cmd=%.12s, Steer=%03.2f, Throttle=%03.2f, Brake=%02.1f, "
                             "Speed=%04.2f" %
                             (data_indices[i], training_configs.cBranchesOutList[int(high_lvl_cmds[i])], steers[i],
                              throttles[i], brakes[i], speeds[i]))
            elif train_data == 'il_dataset':
                fig.suptitle("Data sample index=%d, High_lvl_cmd=%.12s, Steer=%03.2f, Throttle=%03.2f, Brake=%02.1f, "
                             "Speed=%04.2f" %
                             (data_indices[i], training_configs.cBranchesOutList[int(high_lvl_cmds[i])], steers[i],
                              throttles[i], brakes[i], speeds[i]))

            # plt.pause(delay)
            plt.draw()
            plt.cla()
            # plt.show(block=False)
            # plt.waitforbuttonpress()
            # import pdb
            # pdb.set_trace()

            # plt.show(block=False)
            if train_data == 'auc2':
                fig.savefig(store_folder + '/%s.png' % data_indices[i].replace('/', '#'), dpi=fig.dpi)
            elif train_data == 'il_dataset':
                fig.savefig(store_folder + '/sample%i.png' % data_indices[i], dpi=fig.dpi)
            x = 10


# ------------------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    # Set seeds
    os.environ['PYTHONHASHSEED'] = str(1)
    random.seed(2)
    np.random.seed(3)

    # training configurations & data loader
    training_configs = Training_configs()
    data_handler = Data(training_configs)
    iterate_batches_train, iterate_batches_val = data_handler.get_iterators()

    # Compare two models
    compare_two_models = False
    if compare_two_models:
        tf_config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        tf_config.gpu_options.visible_device_list = str(0)
        tf_config.gpu_options.per_process_gpu_memory_fraction = 0.3
        tf.reset_default_graph()
        sessGraph1 = tf.Graph()
        sessGraph2 = tf.Graph()
        with tf.device('/gpu:' + str(0)):
            with sessGraph1.as_default():
                sess = tf.Session(graph=sessGraph1, config=tf_config)
                with sess.as_default():
                    # Build the model
                    netTensors = create_network(training_configs, is_training=True)
                    # Initialize variables
                    saver = tf.train.Saver(write_version=saver_pb2.SaverDef.V2, max_to_keep=0)
                    sess.run(tf.global_variables_initializer())
                    # Load trained parameters
                    path = "/media/cai1/data/heraqi/int-end-to-end-ad/models/F3F1E4_CIL_data_41/epoch_1/model.ckpt"
                    saver.restore(sess, path)
                    print("Loaded model %s" % path)
                    graph_1 = tf.get_default_graph().as_graph_def()
                    weights1 = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]  #  if v.name.endswith('weights:0')
                    weights1_vals = sess.run(weights1)
            with sessGraph2.as_default():
                sess = tf.Session(graph=sessGraph2, config=tf_config)
                with sess.as_default():
                    # Build the model
                    netTensors = create_network(training_configs, is_training=True)
                    # Initialize variables
                    saver = tf.train.Saver(write_version=saver_pb2.SaverDef.V2, max_to_keep=0)
                    sess.run(tf.global_variables_initializer())
                    # Load trained parameters
                    path = "/media/cai1/data/heraqi/int-end-to-end-ad/models/F3F1E4_CIL_data_42/epoch_1/model.ckpt"
                    saver.restore(sess, path)
                    print("Loaded model %s" % path)
                    graph_2 = tf.get_default_graph().as_graph_def()
                    weights2 = [v for v in  tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]  # if v.name.endswith('weights:0')
                    weights2_vals = sess.run(weights2)
        # Model differences
        from tensorflow.python import pywrap_tensorflow
        diff = pywrap_tensorflow.EqualGraphDefWrapper(graph_1.SerializeToString(), graph_2.SerializeToString())
        print(diff)
        # Weights differences
        (weights1_vals[0] == weights2_vals[0]).all()
        (weights1_vals[0][-1][-1][-1][-1] == weights2_vals[0]).all()
        exit()

    # Seen GPU's and memory_fraction used are selected
    # tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    tf_config = tf.ConfigProto()
    # tf_config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)  # Might needed to make things reproducable if needed, but trains on CPU
    tf_config.gpu_options.visible_device_list = str(training_configs.gpu_id)
    tf_config.gpu_options.per_process_gpu_memory_fraction = training_configs.gpu_memory_fraction
    tf.reset_default_graph()
    sessGraph = tf.Graph()

    with tf.device('/gpu:' + str(training_configs.gpu_id)):
        with sessGraph.as_default():
            tf.set_random_seed(4)  # Set seed
            sess = tf.Session(graph=sessGraph, config=tf_config)
            with sess.as_default():
                # Build the model
                netTensors = create_network(training_configs, is_training=True)
                print(netTensors['network_branches'])

                # Initialize variables
                print('Initialize Variables in the Graph ...')
                saver = tf.train.Saver(write_version=saver_pb2.SaverDef.V2, max_to_keep=0)
                sess.run(tf.global_variables_initializer())

                # merge all summaries into a single op
                merged_summary_op = tf.summary.merge_all()

                # Load trained parameters
                if (training_configs.load_model):  # restore trained parameters
                    path = training_configs.model_to_load + "/model.ckpt"
                    saver.restore(sess, path)
                    print("Loaded model %s" % path)

                # op to write logs to Tensorboard
                summary_writer = tf.summary.FileWriter('./logs', graph=sessGraph)

                # Start training
                print('Start Training process ...')
                steps = 0
                losses_file = open(training_configs.modelsPath + "/" +
                                   training_configs.experiment_MDL_hash + '/losses.txt', 'w')
                for epoch in range(training_configs.no_epochs):
                    tStartEpoch = time.time()
                    print("  Epoch:", epoch+1)
                    mini_batch = 0
                    for data, num_of_batches in iterate_batches_train(shuffle_data=True,
                                                                      augment_data=training_configs.augment_data):
                        images_cam_1, images_cam_2, images_cam_3, images_pgm, steers, throttles, brakes, speeds, \
                        high_lvl_cmds, data_indices = data  # data_indices is just returned for debugging

                        # Visualize batch training data
                        visualize = False
                        if mini_batch == 0 and epoch == 0 and visualize:  # set which minibatch to start visualizing
                            visualize_data(data, training_configs.train_data, display_shuffled_samples=False, delay=5)

                        # Prepare feedDict
                        visualize = False
                        if visualize:
                            for img in images_cam_1:
                                plt.imshow(img)
                                print('min=' + str(np.min(img)) + ', max=' + str(np.max(img)))
                                plt.show(block=False)
                                plt.waitforbuttonpress()
                        feedDict = {netTensors['inputs'][0]: images_cam_1,
                                    netTensors['inputs'][1]: speeds.reshape((-1, 1)),
                                    netTensors['inputs'][2]: images_cam_2,
                                    netTensors['inputs'][3]: images_cam_3,
                                    netTensors['inputs'][4]: images_pgm,
                                    netTensors['high_lvl_cmds']: high_lvl_cmds,
                                    # netTensors['high_lvl_cmds']: np.zeros(training_configs.batch_size),  # To train a single branch for now!
                                    netTensors['drop_outs']: training_configs.dropout_vec,
                                    netTensors['LR']: training_configs.learning_rate,
                                    netTensors['targets'][0]: np.hstack([steers.reshape((-1, 1)),
                                                                         throttles.reshape((-1, 1)),
                                                                         brakes.reshape((-1, 1))]),
                                    netTensors['targets'][1]: speeds.reshape((-1, 1))}

                        # Train
                        contSolver = netTensors['optimizers']
                        contLoss = netTensors['losses']

                        # print(images_cam_1)  # For debugging

                        pr = netTensors['print']
                        _, loss_value = sess.run([contSolver, contLoss], feed_dict=feedDict)
                        # print(merged_summary_op)
                        # summary = merged_summary_op.eval(feed_dict=feedDict)

                        # update progress steps
                        mini_batch += 1
                        steps += 1

                        # Print training progress
                        if mini_batch % 1 == 0:  # print each how many minibatches
                            # summary_writer.add_summary(summary, epoch * num_images/batchSize + j)
                            if training_configs.single_branch_per_minibatch:
                                print("    Train::: Epoch: %d & Minibatch: %d/%d (Step: %d), Train_Batch_Loss: %g, Branch "
                                      "Trained: %s, Learning Rate: %g" %
                                      (epoch+1, mini_batch, num_of_batches, steps, loss_value,
                                       training_configs.cBranchesOutList[int(high_lvl_cmds[0])],
                                       training_configs.learning_rate))
                            else:
                                print("    Train::: Epoch: %d & Minibatch: %d/%d (Step: %d), Train_Batch_Loss: %g, "
                                      "Learning Rate: %g" %
                                      (epoch + 1, mini_batch, num_of_batches, steps, loss_value,
                                       training_configs.learning_rate))

                        # Debug weights
                        debug_weights = False
                        if debug_weights:
                            weights = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]  # if v.name.endswith('weights:0')
                            weights_vals = sess.run(weights)
                            print(weights_vals[0][0][0][0])
                            exit()

                    # Validate & Save model per epochs
                    tStopEpoch = time.time()
                    print("  Epoch Time Cost:", round(tStopEpoch - tStartEpoch, 2), "s")
                    if (epoch + 1) % training_configs.save_each_epoch == 0:
                        train_loss, val_loss = validate_model(iterate_batches_train, iterate_batches_val)
                        losses_file.write("Steps: %d, Epoch: %d, Minibatch: %d, Train Loss: %g, Validation Loss: %g, "
                                          "Learning Rate: %g\n" %
                                          (steps, epoch + 1, mini_batch, train_loss, val_loss,
                                           training_configs.learning_rate))
                        losses_file.flush()
                        print('Save Checkpoint ...')
                        dir = training_configs.modelsPath + "/" + training_configs.experiment_MDL_hash + "/epoch_" + \
                              str(epoch + 1)
                        if not os.path.exists(dir):
                            os.makedirs(dir)
                        checkpoint_path = dir + "/model.ckpt"
                        filename = saver.save(sess, checkpoint_path)
                        print("  Model saved in file: %s" % filename)
                        
                    # Divide learning rate
                    if epoch % training_configs.divide_learning_rate_after_epochs == 0:
                        training_configs.learning_rate = training_configs.learning_rate / \
                                                         training_configs.divide_learning_rate_by
                        if training_configs.learning_rate < training_configs.learning_rate_min:
                            training_configs.learning_rate = training_configs.learning_rate_min


