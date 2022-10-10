from future.builtins import range
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import pandas
import glob
import h5py
from imgaug import augmenters as iaa
import json
from tqdm import tqdm
import pickle


# ------------------------------------------------------------------------------
# CARLA data class
# ------------------------------------------------------------------------------
class Data:
    """
    images are generated BGR, because they are read using cv2.imread
    """
    def __init__(self, training_configs):
        self.datasetDirTrain = training_configs.datasetDirTrain
        self.datasetDirVal = training_configs.datasetDirVal
        self.meta_data_filename = training_configs.meta_data_filename
        self.train_data = training_configs.train_data
        self.single_branch_per_minibatch = training_configs.single_branch_per_minibatch
        self.batch_size = training_configs.batch_size
        self.nbr_steering_bins = training_configs.nbr_steering_bins
        self.no_epochs = training_configs.no_epochs
        self.branchConfigs = training_configs.branchConfigs
        self.cBranchesOutList = training_configs.cBranchesOutList
        self.meta_data_filename = training_configs.meta_data_filename
        
        self.augmentation_ratio = training_configs.augmentation_ratio
        self.seq_augmentation = self.create_augmenter()
        
        if training_configs.train_data == "auc1":
            self.cam1_image_files_train, self.cam2_image_files_train, self.cam3_image_files_train, \
            self.pgm_image_files_train, self.high_lvl_cmds_train, self.steers_train, self.throttles_train, \
            self.brakes_train, self.speeds_train = self.load_auc1_dataset_metadata(dataset_dir=self.datasetDirTrain)

            self.cam1_image_files_val, self.cam2_image_files_val, self.cam3_image_files_val, \
            self.pgm_image_files_val, self.high_lvl_cmds_val, self.steers_val, self.throttles_val, \
            self.brakes_val, self.speeds_val = self.load_auc1_dataset_metadata(dataset_dir=self.datasetDirVal)
        elif training_configs.train_data == "auc2":
            fetch = False  # Should happens only once if dataset changes
            if fetch:  # Load metadata from harddesk and store it
                self.cam1_image_files_train, self.cam2_image_files_train, self.cam3_image_files_train, \
                self.pgm_image_files_train, self.high_lvl_cmds_train, self.steers_train, self.throttles_train, \
                self.brakes_train, self.speeds_train = self.load_auc2_dataset_metadata(dataset_dir=self.datasetDirTrain)

                self.cam1_image_files_val, self.cam2_image_files_val, self.cam3_image_files_val, \
                self.pgm_image_files_val, self.high_lvl_cmds_val, self.steers_val, self.throttles_val, \
                self.brakes_val, self.speeds_val = self.load_auc2_dataset_metadata(dataset_dir=self.datasetDirVal)
                
                # Save auc2 dataset metadata
                print("Saving AUC2 dataset metadata")
                with open('auc2dataset_metadata_train.pickle', 'wb') as f:
                    pickle.dump([
                        self.cam1_image_files_train, self.cam2_image_files_train, self.cam3_image_files_train, \
                        self.pgm_image_files_train, self.high_lvl_cmds_train, self.steers_train, self.throttles_train, \
                        self.brakes_train, self.speeds_train], f)
                with open('auc2dataset_metadata_val.pickle', 'wb') as f:
                    pickle.dump([
                        self.cam1_image_files_val, self.cam2_image_files_val, self.cam3_image_files_val, \
                        self.pgm_image_files_val, self.high_lvl_cmds_val, self.steers_val, self.throttles_val, \
                        self.brakes_val, self.speeds_val], f)
            else:  # Load auc2 dataset metadata
                print("Loading AUC2 dataset metadata")
                with open('auc2dataset_metadata_train.pickle', 'rb') as f:
                    self.cam1_image_files_train, self.cam2_image_files_train, self.cam3_image_files_train, \
                    self.pgm_image_files_train, self.high_lvl_cmds_train, self.steers_train, self.throttles_train, \
                    self.brakes_train, self.speeds_train = pickle.load(f)
                with open('auc2dataset_metadata_val.pickle', 'rb') as f:
                    self.cam1_image_files_val, self.cam2_image_files_val, self.cam3_image_files_val, \
                    self.pgm_image_files_val, self.high_lvl_cmds_val, self.steers_val, self.throttles_val, \
                    self.brakes_val, self.speeds_val = pickle.load(f)
            
        elif training_configs.train_data == "il_dataset":
            self.read_IL_dataset()
        else:
            raise ValueError('training_configs.train_data is not set to a proper value!')

        # Print data amount per branch
        data_lengths = [0, 0, 0, 0]
        for cmd in range(4):
            data_lengths[cmd] = len([i for i in range(len(self.high_lvl_cmds_train))
                                     if (self.high_lvl_cmds_train[i] == cmd)])
        print("  Training data amount per command: %s: %d, %s: %d, %s: %d, %s: %d" %
              (self.cBranchesOutList[0], data_lengths[0], self.cBranchesOutList[1], data_lengths[1],
               self.cBranchesOutList[2], data_lengths[2], self.cBranchesOutList[3], data_lengths[3]))
        for cmd in range(4):
            data_lengths[cmd] = len([i for i in range(len(self.high_lvl_cmds_val))
                                     if (self.high_lvl_cmds_val[i] == cmd)])
        print("  Validation data amount per command: %s: %d, %s: %d, %s: %d, %s: %d" %
              (self.cBranchesOutList[0], data_lengths[0], self.cBranchesOutList[1], data_lengths[1],
               self.cBranchesOutList[2], data_lengths[2], self.cBranchesOutList[3], data_lengths[3]))

    def get_iterators(self):
        if self.train_data == "auc1":
            if self.single_branch_per_minibatch:
                return self.iterate_batches_auc1_dataset_train_singlebranch, \
                       self.iterate_batches_auc1_dataset_val_singlebranch
            else:
                return self.iterate_batches_auc1_dataset_train, \
                       self.iterate_batches_auc1_dataset_val
        elif self.train_data == "auc2":
            return self.iterate_batches_auc2_dataset_train, \
                   self.iterate_batches_auc2_dataset_val
        elif self.train_data == "il_dataset":
            if self.single_branch_per_minibatch:
                return self.iterate_batches_IL_dataset_train_singlebranch, \
                       self.iterate_batches_IL_dataset_val_singlebranch
            else:
                return self.iterate_batches_IL_dataset_train, \
                       self.iterate_batches_IL_dataset_val
        else:
            raise ValueError('training_configs.train_data is not set to a proper value!')

    def create_augmenter(self):
        # Normal Augemnation
        '''st = lambda aug: iaa.Sometimes(0.4, aug)  # 40% of images to be augmented
        oc = lambda aug: iaa.Sometimes(0.3, aug)  # 30% of images to be augmented
        rl = lambda aug: iaa.Sometimes(0.09, aug)  # 9% of images to be augmented'''
        # Aggressive Aigemenation
        st = lambda aug: iaa.Sometimes(0.6, aug)  # 60% of images to be augmented
        oc = lambda aug: iaa.Sometimes(0.45, aug)  # 45% of images to be augmented
        rl = lambda aug: iaa.Sometimes(0.14, aug)  # 14% of images to be augmented
        seq = iaa.Sequential([
            rl(iaa.GaussianBlur((0, 1.5))),  # blur images with a sigma between 0 and 1.5
            rl(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05), per_channel=0.5)),  # add gaussian noise to images
            oc(iaa.Dropout((0.0, 0.10), per_channel=0.5)),  # randomly remove up to X% of the pixels
            oc(iaa.CoarseDropout((0.0, 0.10), size_percent=(0.08, 0.2), per_channel=0.5)),
            # randomly remove up to X% of the pixels
            oc(iaa.Add((-40, 40), per_channel=0.5)),  # change brightness of images (by -X to Y of original value)
            st(iaa.Multiply((0.10, 2.5), per_channel=0.2)),  # change brightness of images (X-Y% of original value)
            rl(iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)),  # improve or worsen the contrast
            rl(iaa.Grayscale((0.0, 1))), # put grayscale  # TODO: put or remove?
        ], random_order=True)  # order is shuffled each time with equal probability
        return seq

    # AUC dataset 1 is stored as BGR
    def load_auc1_dataset_metadata(self, dataset_dir):
        # Parse training episodes metadata
        print('Reading AUC Dataset from: ' + dataset_dir + ' ...')
        images_filenames = []  # list of all data images formatted like: episode_filename
        high_lvl_cmds = []
        steers = []
        throttles = []
        brakes = []
        speeds = []
        episode = 0
        current_episode_dir = dataset_dir + "/%i" % episode
        while os.path.isdir(current_episode_dir):
            metadata_csv_file_path = current_episode_dir + self.meta_data_filename
            data = pandas.read_csv(metadata_csv_file_path)

            # Frames where the car is not stopped & skip some frames in episode start (hardcoded number)
            good_frames_idx = [i for i in range(10, len(data['high_lvl_cmd'])) if data['high_lvl_cmd'][i] != -1]

            images_filenames.extend([str(episode) + '_' + '%08d.png' % data['step'][i] for i in good_frames_idx])
            high_lvl_cmds.extend([data['high_lvl_cmd'][i] for i in good_frames_idx])
            steers.extend([data['steer'][i] for i in good_frames_idx])
            throttles.extend([data['throttle'][i] for i in good_frames_idx])
            brakes.extend([data['brake'][i] for i in good_frames_idx])
            # (forward_speed is in m/s) normalize by 90km/h=25m/s to be from 0 to 1
            speeds.extend([data['forward_speed'][i]/25.0 for i in good_frames_idx])

            episode += 1
            current_episode_dir = dataset_dir + "/%i" % episode

        visualize = False
        if visualize:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(speeds)
            plt.show()

        # Convert filenames to absolute paths
        cam1_image_files = [dataset_dir + '/' + img.split('_')[0] + '/CameraRGB_centre/' + img.split('_')[1]
                            for img in images_filenames]
        cam2_image_files = [dataset_dir + '/' + img.split('_')[0] + '/CameraRGB_right/' + img.split('_')[1]
                            for img in images_filenames]
        cam3_image_files = [dataset_dir + '/' + img.split('_')[0] + '/CameraRGB_left/' + img.split('_')[1]
                            for img in images_filenames]
        pgm_image_files = [dataset_dir + '/' + img.split('_')[0] + '/LiDAR_PGM/' + img.split('_')[1]
                           for img in images_filenames]

        return np.array(cam1_image_files), np.array(cam2_image_files), np.array(cam3_image_files), \
               np.array(pgm_image_files), np.array(high_lvl_cmds), np.array(steers), np.array(throttles),\
               np.array(brakes), np.array(speeds)

    # Provides images are from 0 to 1, this should be respected during inference time
    # Minibatches contain an equal number of samples with each command
    # Images are resided to 88x200 (height x width) to be like IL dataset
    def iterate_batches_auc1_dataset_train(self, shuffle_data=True, augment_data=False, batch_size=-1,
                                          sample_percent=100):
        if batch_size == -1:
            batch_size = self.batch_size

        # Shuffle training data
        shuffled_idx = np.arange(len(self.steers_train))
        if shuffle_data:
            np.random.shuffle(shuffled_idx)
        shuffled_idx = shuffled_idx[0:int(len(shuffled_idx) * sample_percent / 100.)]  # Sample from data

        # Split data by High Level Command
        bins_idx = [[] for _ in range(4)]
        for cmd in range(4):
            bins_idx[cmd] = [i for i in shuffled_idx if (self.high_lvl_cmds_train[i] == cmd)]
            
        print("Training data samples count: " + str(len(self.high_lvl_cmds_train)))
        print("  Go Right high-level command samples count: " + str(len(bins_idx[0])))
        print("  Go Left high-level command samples count: " + str(len(bins_idx[3])))
        print("  Go Straight high-level command samples count: " + str(len(bins_idx[1])))
        print("  Follow Lane high-level command samples count: " + str(len(bins_idx[2])))

        # Iterate the epoch batches
        epoch_size_min_bin = min([len(bins_idx[i]) for i in range(len(bins_idx))])
        stride = int(batch_size / 4)  # batch_size should be multiple of 4 commands
        num_of_batches = len(range(0, epoch_size_min_bin, stride))
        for batch_start_id in range(0, epoch_size_min_bin, stride):
            to = min(epoch_size_min_bin, batch_start_id + stride)

            data_indices = []
            batch_high_lvl_cmds = []
            batch_steers = []
            batch_throttles = []
            batch_brakes = []
            batch_speeds = []
            batch_images_cam_1 = []
            batch_images_cam_2 = []
            batch_images_cam_3 = []
            batch_images_pgm = []
            for cmd in range(4):
                indices = bins_idx[cmd][batch_start_id:to]

                data_indices.extend(indices)
                batch_high_lvl_cmds.extend(self.high_lvl_cmds_train[indices])
                batch_steers.extend(self.steers_train[indices])
                batch_throttles.extend(self.throttles_train[indices])
                batch_brakes.extend(self.brakes_train[indices])
                batch_speeds.extend(self.speeds_train[indices])
                batch_images_cam_1.extend([cv2.resize(cv2.imread(img), (200, 88)) for img in
                                           self.cam1_image_files_train[indices]])
                batch_images_cam_2.extend([cv2.resize(cv2.imread(img), (200, 88)) for img in
                                           self.cam2_image_files_train[indices]])
                batch_images_cam_3.extend([cv2.resize(cv2.imread(img), (200, 88)) for img in
                                           self.cam3_image_files_train[indices]])
                batch_images_pgm.extend([cv2.resize(cv2.imread(img, cv2.IMREAD_GRAYSCALE), (200, 88)) for
                                         img in self.pgm_image_files_train[indices]])

            # Augment data
            if augment_data:
                idx_to_augment = np.random.randint(low=0, high=len(batch_images_cam_1),
                                                   size=int(len(batch_images_cam_1) * self.augmentation_ratio))
                augmented_images = self.seq_augmentation.augment_images(np.array([batch_images_cam_1[i] for i in
                                                                             idx_to_augment], dtype=np.uint8))
                i = 0
                for id in idx_to_augment:
                    batch_images_cam_1[id] = augmented_images[i]
                    i += 1

                idx_to_augment = np.random.randint(low=0, high=len(batch_images_cam_2),
                                                   size=int(len(batch_images_cam_2) * self.augmentation_ratio))
                augmented_images = self.seq_augmentation.augment_images(np.array([batch_images_cam_2[i] for i in
                                                                             idx_to_augment], dtype=np.uint8))
                i = 0
                for id in idx_to_augment:
                    batch_images_cam_2[id] = augmented_images[i]
                    i += 1

                idx_to_augment = np.random.randint(low=0, high=len(batch_images_cam_3),
                                                   size=int(len(batch_images_cam_3) * self.augmentation_ratio))
                augmented_images = self.seq_augmentation.augment_images(np.array([batch_images_cam_3[i] for i in
                                                                             idx_to_augment], dtype=np.uint8))
                i = 0
                for id in idx_to_augment:
                    batch_images_cam_3[id] = augmented_images[i]
                    i += 1

                # Don't augment LiDAR PGM
                '''idx_to_augment = np.random.randint(low=0, high=len(batch_images_pgm),
                                                   size=int(len(batch_images_pgm) * self.augmentation_ratio))
                augmented_images = self.seq_augmentation.augment_images(np.array([batch_images_pgm[i] for i in
                                                                             idx_to_augment], dtype=np.uint8))
                i = 0
                for id in idx_to_augment:
                    batch_images_pgm[id] = augmented_images[i]
                    i += 1'''

            # High level command: 0:'Go Right', 1:'Go Straight', 2:'Follow Lane', 3:'Go Left'
            yield (np.array(batch_images_cam_1) / 255., np.array(batch_images_cam_2) / 255.,
                   np.array(batch_images_cam_3) / 255., np.array(batch_images_pgm) / 255., np.array(batch_steers),
                   np.array(batch_throttles), np.array(batch_brakes), np.array(batch_speeds),
                   np.array(batch_high_lvl_cmds), np.array(data_indices)), num_of_batches

    # Provides images are from 0 to 1, this should be respected during inference time
    # Minibatches contain an equal number of samples with each command
    # Images are resided to 88x200 (height x width) to be like IL dataset
    def iterate_batches_auc1_dataset_val(self, shuffle_data=True, augment_data=False, batch_size=-1,
                                        sample_percent=100):
        if batch_size == -1:
            batch_size = self.batch_size

        # Shuffle validation data
        shuffled_idx = np.arange(len(self.steers_val))
        if shuffle_data:
            np.random.shuffle(shuffled_idx)
        shuffled_idx = shuffled_idx[0:int(len(shuffled_idx) * sample_percent / 100.)]  # Sample from data

        # Split data by High Level Command
        bins_idx = [[] for _ in range(4)]
        for cmd in range(4):
            bins_idx[cmd] = [i for i in shuffled_idx if (self.high_lvl_cmds_val[i] == cmd)]

        # Iterate the epoch batches
        epoch_size_min_bin = min([len(bins_idx[i]) for i in range(len(bins_idx))])
        stride = int(batch_size / 4)  # batch_size should be multiple of 4 commands
        num_of_batches = len(range(0, epoch_size_min_bin, stride))
        for batch_start_id in range(0, epoch_size_min_bin, stride):
            to = min(epoch_size_min_bin, batch_start_id + stride)

            data_indices = []
            batch_high_lvl_cmds = []
            batch_steers = []
            batch_throttles = []
            batch_brakes = []
            batch_speeds = []
            batch_images_cam_1 = []
            batch_images_cam_2 = []
            batch_images_cam_3 = []
            batch_images_pgm = []
            for cmd in range(4):
                indices = bins_idx[cmd][batch_start_id:to]

                data_indices.extend(indices)
                batch_high_lvl_cmds.extend(self.high_lvl_cmds_val[indices])
                batch_steers.extend(self.steers_val[indices])
                batch_throttles.extend(self.throttles_val[indices])
                batch_brakes.extend(self.brakes_val[indices])
                batch_speeds.extend(self.speeds_val[indices])
                batch_images_cam_1.extend([cv2.resize(cv2.imread(img), (200, 88)) for img in
                                           self.cam1_image_files_val[indices]])
                batch_images_cam_2.extend([cv2.resize(cv2.imread(img), (200, 88)) for img in
                                           self.cam2_image_files_val[indices]])
                batch_images_cam_3.extend([cv2.resize(cv2.imread(img), (200, 88)) for img in
                                           self.cam3_image_files_val[indices]])
                batch_images_pgm.extend([cv2.resize(cv2.imread(img, cv2.IMREAD_GRAYSCALE), (200, 88)) for
                                         img in self.pgm_image_files_val[indices]])
            
            # High level command: 0:'Go Right', 1:'Go Straight', 2:'Follow Lane', 3:'Go Left'
            yield (np.array(batch_images_cam_1) / 255., np.array(batch_images_cam_2) / 255.,
                   np.array(batch_images_cam_3) / 255., np.array(batch_images_pgm) / 255., np.array(batch_steers),
                   np.array(batch_throttles), np.array(batch_brakes), np.array(batch_speeds),
                   np.array(batch_high_lvl_cmds), np.array(data_indices)), num_of_batches

    # AUC dataset 2 is stored as RGB
    def load_auc2_dataset_metadata(self, dataset_dir):
        # Parse training episodes metadata
        print('Reading AUC Dataset from: ' + dataset_dir + ' ...')
        cam1_image_files = []
        cam2_image_files = []
        cam3_image_files = []
        pgm_image_files = []
        high_lvl_cmds = []
        steers = []
        throttles = []
        brakes = []
        speeds = []

        episodes = [os.path.join(dataset_dir, o) for o in os.listdir(dataset_dir) if
                    os.path.isdir(os.path.join(dataset_dir, o))]
        for current_episode_dir in tqdm(episodes):
            '''metadata_json_file_path = current_episode_dir + '/' + self.meta_data_filename
            with open(metadata_json_file_path) as f:
                meta_data = json.load(f)'''

            files = [f for f in os.listdir(current_episode_dir) if f.startswith('measurements_')]
            for i in range(len(files)):
                json_file = current_episode_dir + '/' + 'measurements_' + str(i).zfill(5) + '.json'
                if not os.path.isfile(json_file):
                    print('JSON file does not exist: %s' % f)
                    continue
                with open(json_file) as f:
                    try:
                        data = json.load(f)
                    except json.decoder.JSONDecodeError:
                        print('Could not load JSON file: %s' % f)
                        continue

                cam1_image_files.append(current_episode_dir + '/' + 'CentralRGB_' + str(i).zfill(5) + '.png')
                cam2_image_files.append(current_episode_dir + '/' + 'RightRGB_' + str(i).zfill(5) + '.png')
                cam3_image_files.append(current_episode_dir + '/' + 'LeftRGB_' + str(i).zfill(5) + '.png')
                pgm_image_files.append(current_episode_dir + '/' + 'Lidar32_' + str(i).zfill(5) + '.png')

                # high_lvl_cmd = 2:LANE_FOLLOW, 0:REACH_GOAL, 3:TURN_LEFT, 4:TURN_RIGHT, 5:GO_STRAIGHT
                high_lvl_cmds.append(data['directions'])
                steers.append(data['steer'])
                throttles.append(data['throttle'])
                brakes.append(data['brake'])
                # (forward_speed is in m/s) normalize by 90km/h=25m/s to be from 0 to 1
                if 'forwardSpeed' in data['playerMeasurements']:
                    speeds.append(data['playerMeasurements']['forwardSpeed'] / 25.0)
                else:
                    speeds.append(0)

        # replace REACH_GOAL by LANE_FOLLOW and make commands from 0 to 3:
        brakes[high_lvl_cmds == 0.0] = 0.0  # Because on REACH_GOAL brake could be is 1
        high_lvl_cmds[high_lvl_cmds == 0.0] = 2.0

        visualize = False
        if visualize:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(high_lvl_cmds)
            plt.show()

        # Convert to: high_lvl_cmd = 0:TURN_RIGHT, 1:GO_STRAIGHT, 2:LANE_FOLLOW, 3:TURN_LEFT
        return np.array(cam1_image_files), np.array(cam2_image_files), np.array(cam3_image_files), \
               np.array(pgm_image_files), np.array(high_lvl_cmds) % 4, np.array(steers), np.array(throttles), \
               np.array(brakes), np.array(speeds)

    # Provides images are from 0 to 1, this should be respected during inference time
    # Minibatches contain an equal number of samples with each command
    # Images are resided to 88x200 (height x width) to be like IL dataset
    def iterate_batches_auc2_dataset_train(self, shuffle_data=True, augment_data=False, batch_size=-1,
                                           sample_percent=100):
        if batch_size == -1:
            batch_size = self.batch_size

        # Shuffle training data
        shuffled_idx = np.arange(len(self.steers_train))
        if shuffle_data:
            np.random.shuffle(shuffled_idx)
        shuffled_idx = shuffled_idx[0:int(len(shuffled_idx) * sample_percent / 100.)]  # Sample from data

        # Split data by High Level Command
        bins_idx = [[] for _ in range(4)]
        for cmd in range(4):
            bins_idx[cmd] = [i for i in shuffled_idx if (self.high_lvl_cmds_train[i] == cmd)]

        '''print("Training data samples count: " + str(len(self.high_lvl_cmds_train)))
        print("  Go Right high-level command samples count: " + str(len(bins_idx[0])))
        print("  Go Left high-level command samples count: " + str(len(bins_idx[3])))
        print("  Go Straight high-level command samples count: " + str(len(bins_idx[1])))
        print("  Follow Lane high-level command samples count: " + str(len(bins_idx[2])))'''

        # Iterate the epoch batches
        epoch_size_min_bin = min([len(bins_idx[i]) for i in range(len(bins_idx))])
        stride = int(batch_size / 4)  # batch_size should be multiple of 4 commands
        num_of_batches = len(range(0, epoch_size_min_bin, stride))
        for batch_start_id in range(0, epoch_size_min_bin, stride):
            to = min(epoch_size_min_bin, batch_start_id + stride)

            data_indices = []
            batch_high_lvl_cmds = []
            batch_steers = []
            batch_throttles = []
            batch_brakes = []
            batch_speeds = []
            batch_images_cam_1 = []
            batch_images_cam_2 = []
            batch_images_cam_3 = []
            batch_images_pgm = []
            for cmd in range(4):
                indices = bins_idx[cmd][batch_start_id:to]

                data_indices.extend(self.cam1_image_files_train[indices])
                batch_high_lvl_cmds.extend(self.high_lvl_cmds_train[indices])
                batch_steers.extend(self.steers_train[indices])
                batch_throttles.extend(self.throttles_train[indices])
                batch_brakes.extend(self.brakes_train[indices])
                batch_speeds.extend(self.speeds_train[indices])
                batch_images_cam_1.extend([cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
                                           for img in self.cam1_image_files_train[indices]])
                batch_images_cam_2.extend([cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
                                           for img in self.cam2_image_files_train[indices]])
                batch_images_cam_3.extend([cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
                                           for img in self.cam3_image_files_train[indices]])
                batch_images_pgm.extend([cv2.imread(img, cv2.IMREAD_GRAYSCALE) for
                                         img in self.pgm_image_files_train[indices]])

            # Augment data
            if augment_data:
                idx_to_augment = np.random.randint(low=0, high=len(batch_images_cam_1),
                                                   size=int(len(batch_images_cam_1) * self.augmentation_ratio))
                augmented_images = self.seq_augmentation.augment_images(np.array([batch_images_cam_1[i] for i in
                                                                             idx_to_augment], dtype=np.uint8))
                i = 0
                for id in idx_to_augment:
                    batch_images_cam_1[id] = augmented_images[i]
                    i += 1

                idx_to_augment = np.random.randint(low=0, high=len(batch_images_cam_2),
                                                   size=int(len(batch_images_cam_2) * self.augmentation_ratio))
                augmented_images = self.seq_augmentation.augment_images(np.array([batch_images_cam_2[i] for i in
                                                                             idx_to_augment], dtype=np.uint8))
                i = 0
                for id in idx_to_augment:
                    batch_images_cam_2[id] = augmented_images[i]
                    i += 1

                idx_to_augment = np.random.randint(low=0, high=len(batch_images_cam_3),
                                                   size=int(len(batch_images_cam_3) * self.augmentation_ratio))
                augmented_images = self.seq_augmentation.augment_images(np.array([batch_images_cam_3[i] for i in
                                                                             idx_to_augment], dtype=np.uint8))
                i = 0
                for id in idx_to_augment:
                    batch_images_cam_3[id] = augmented_images[i]
                    i += 1

                # Don't augment LiDAR PGM
                '''idx_to_augment = np.random.randint(low=0, high=len(batch_images_pgm),
                                                   size=int(len(batch_images_pgm) / 2))
                augmented_images = self.seq_augmentation.augment_images(np.array([batch_images_pgm[i] for i in
                                                                             idx_to_augment], dtype=np.uint8))
                i = 0
                for id in idx_to_augment:
                    batch_images_pgm[id] = augmented_images[i]
                    i += 1'''

            # High level command: 0:'Go Right', 1:'Go Straight', 2:'Follow Lane', 3:'Go Left'
            yield (np.array(batch_images_cam_1) / 255., np.array(batch_images_cam_2) / 255.,
                   np.array(batch_images_cam_3) / 255., np.array(batch_images_pgm) / 255., np.array(batch_steers),
                   np.array(batch_throttles), np.array(batch_brakes), np.array(batch_speeds),
                   np.array(batch_high_lvl_cmds), data_indices), num_of_batches

    # Provides images are from 0 to 1, this should be respected during inference time
    # Minibatches contain an equal number of samples with each command
    # Images are resided to 88x200 (height x width) to be like IL dataset
    def iterate_batches_auc2_dataset_val(self, shuffle_data=True, augment_data=False, batch_size=-1,
                                         sample_percent=100):
        if batch_size == -1:
            batch_size = self.batch_size

        # Shuffle validation data
        shuffled_idx = np.arange(len(self.steers_val))
        if shuffle_data:
            np.random.shuffle(shuffled_idx)
        shuffled_idx = shuffled_idx[0:int(len(shuffled_idx) * sample_percent / 100.)]  # Sample from data

        # Split data by High Level Command
        bins_idx = [[] for _ in range(4)]
        for cmd in range(4):
            bins_idx[cmd] = [i for i in shuffled_idx if (self.high_lvl_cmds_val[i] == cmd)]

        # Iterate the epoch batches
        epoch_size_min_bin = min([len(bins_idx[i]) for i in range(len(bins_idx))])
        stride = int(batch_size / 4)  # batch_size should be multiple of 4 commands
        num_of_batches = len(range(0, epoch_size_min_bin, stride))
        for batch_start_id in range(0, epoch_size_min_bin, stride):
            to = min(epoch_size_min_bin, batch_start_id + stride)

            data_indices = []
            batch_high_lvl_cmds = []
            batch_steers = []
            batch_throttles = []
            batch_brakes = []
            batch_speeds = []
            batch_images_cam_1 = []
            batch_images_cam_2 = []
            batch_images_cam_3 = []
            batch_images_pgm = []
            for cmd in range(4):
                indices = bins_idx[cmd][batch_start_id:to]

                data_indices.extend(self.cam1_image_files_val[indices])
                batch_high_lvl_cmds.extend(self.high_lvl_cmds_val[indices])
                batch_steers.extend(self.steers_val[indices])
                batch_throttles.extend(self.throttles_val[indices])
                batch_brakes.extend(self.brakes_val[indices])
                batch_speeds.extend(self.speeds_val[indices])
                batch_images_cam_1.extend([cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
                                           for img in self.cam1_image_files_val[indices]])
                batch_images_cam_2.extend([cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
                                           for img in self.cam2_image_files_val[indices]])
                batch_images_cam_3.extend([cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
                                           for img in self.cam3_image_files_val[indices]])
                batch_images_pgm.extend([cv2.imread(img, cv2.IMREAD_GRAYSCALE) for
                                         img in self.pgm_image_files_val[indices]])

            # High level command: 0:'Go Right', 1:'Go Straight', 2:'Follow Lane', 3:'Go Left'
            yield (np.array(batch_images_cam_1) / 255., np.array(batch_images_cam_2) / 255.,
                   np.array(batch_images_cam_3) / 255., np.array(batch_images_pgm) / 255., np.array(batch_steers),
                   np.array(batch_throttles), np.array(batch_brakes), np.array(batch_speeds),
                   np.array(batch_high_lvl_cmds), data_indices), num_of_batches

    def read_IL_dataset(self):
        datasetTrainFiles_filenames = glob.glob(self.datasetDirTrain + '/*.h5')
        datasetValFiles_filenames = glob.glob(self.datasetDirVal + '/*.h5')

        # read training data
        print('Reading Training IL Data ...')
        self.imgs_train = []
        high_lvl_cmds_train = []
        steers_train = []
        throttles_train = []
        brakes_train = []
        speeds_train = []
        for idx in range(len(datasetTrainFiles_filenames)):  # iterate all files
            try:
                data = h5py.File(datasetTrainFiles_filenames[idx], 'r')
            except:
                print('Could not load file %s' % datasetTrainFiles_filenames[idx])
                continue
            targets = data['targets']
            self.imgs_train.append(data)  # actual images are in data['rgb']
            high_lvl_cmds_train.append(targets[:, 24])
            steers_train.append(targets[:, 0])
            throttles_train.append(targets[:, 1])
            brakes_train.append(targets[:, 2])
            speeds_train.append(targets[:, 10])

        # For debugging, save sample images (Comment this)
        '''import scipy.misc
        for i in range(0, len(high_lvl_cmds_train)):  # iterate all files
            img = self.imgs_train[int(i / 200)]['rgb'][i % 200]
            scipy.misc.imsave('/mnt/sdb1/heraqi/data/int-end-to-end-ad/temp/%d.png'%i, img)
            if i==3021:
                exit()'''
        
        visualize = False
        if visualize:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(np.vstack(speeds_train).flatten())
            plt.show()

        # So it becomes: 0:'Go Right', 1:'Go Straight', 2:'Follow Lane', 3:'Go Left'
        self.high_lvl_cmds_train = np.vstack(high_lvl_cmds_train).flatten().astype(np.int8) % 4
        self.steers_train = np.vstack(steers_train).flatten()
        self.throttles_train = np.vstack(throttles_train).flatten()
        self.brakes_train = np.vstack(brakes_train).flatten()
        self.speeds_train = np.vstack(speeds_train).flatten() / 90.0  # (speeds_train is in km/h) Normalize by 90km/h to be from 0 to 1

        # Visualize for debugging
        visualize = False
        if visualize:
            fig = plt.figure(figsize=(12, 6))
            # idx = np.random.randint(len(datasetFilesTrain_filenames) - 1)
            for i in range(0, len(self.high_lvl_cmds_train)):  # iterate all files
            # for i in [293742]:
                img = self.imgs_train[int(i / 200)]['rgb'][i % 200]
                plt.imshow(img)
                fig.suptitle("Sample Index=%i/%i,\nHigh_lvl_cmd=%.12s, Steer=%03.2f, Throttle=%03.2f, "
                             "Brake=%02.1f, Speed=%04.2f" %
                             (i, len(self.high_lvl_cmds_train),
                              self.cBranchesOutList[int(self.high_lvl_cmds_train[i])], self.steers_train[i],
                              self.throttles_train[i], self.brakes_train[i], self.speeds_train[i]))
                plt.pause(0.02)
                plt.draw()
                input()
                plt.cla()

        print('Reading Validation IL Data ...')
        self.imgs_val = []
        high_lvl_cmds_val = []
        steers_val = []
        throttles_val = []
        brakes_val = []
        speeds_val = []
        for idx in range(len(datasetValFiles_filenames)):  # iterate all files
            try:
                data = h5py.File(datasetValFiles_filenames[idx], 'r')
            except:
                print('Could not load file %s' % datasetValFiles_filenames[idx])
                continue
            targets = data['targets']
            self.imgs_val.append(data)  # actual images are in data['rgb']
            high_lvl_cmds_val.append(targets[:, 24])
            steers_val.append(targets[:, 0])
            throttles_val.append(targets[:, 1])
            brakes_val.append(targets[:, 2])
            speeds_val.append(targets[:, 10])

        # So it becomes: 0:'Go Right', 1:'Go Straight', 2:'Follow Lane', 3:'Go Left'
        self.high_lvl_cmds_val = np.vstack(high_lvl_cmds_val).flatten().astype(np.int8) % 4
        self.steers_val = np.vstack(steers_val).flatten()
        self.throttles_val = np.vstack(throttles_val).flatten()
        self.brakes_val = np.vstack(brakes_val).flatten()
        self.speeds_val = np.vstack(speeds_val).flatten() / 90.0  # (speeds_train is in km/h) Normalize by 90km/h to be from 0 to 1

    # Provides images are from 0 to 1, this should be respected during inference time
    # Minibatches contain an equal number of samples with each command
    def iterate_batches_IL_dataset_train(self, shuffle_data=True, augment_data=False, batch_size=-1,
                                         sample_percent=100):
        if batch_size == -1:
            batch_size = self.batch_size          
            
        # Shuffle training data
        shuffled_idx = np.arange(len(self.steers_train))
        if shuffle_data:
            np.random.shuffle(shuffled_idx)
        shuffled_idx = shuffled_idx[0:int(len(shuffled_idx) * sample_percent / 100.)]  # Sample from data

        # Split data by High Level Command
        bins_idx = [[] for _ in range(4)]
        for cmd in range(4):
            bins_idx[cmd] = [i for i in shuffled_idx if (self.high_lvl_cmds_train[i] == cmd)]

        # Iterate the epoch batches
        epoch_size_min_bin = min([len(bins_idx[i]) for i in range(len(bins_idx))])
        stride = int(batch_size / 4)  # batch_size should be multiple of 4 commands
        num_of_batches = len(range(0, epoch_size_min_bin, stride))
        for batch_start_id in range(0, epoch_size_min_bin, stride):
            to = min(epoch_size_min_bin, batch_start_id + stride)

            data_indices = []  # used to return it for just for debugging issues
            batch_high_lvl_cmds = []
            batch_steers = []
            batch_throttles = []
            batch_brakes = []
            batch_speeds = []
            batch_imgs = []
            for cmd in range(4):
                indices = bins_idx[cmd][batch_start_id:to]

                data_indices.extend(indices)
                batch_high_lvl_cmds.extend(self.high_lvl_cmds_train[indices])
                batch_steers.extend(self.steers_train[indices])
                batch_throttles.extend(self.throttles_train[indices])
                batch_brakes.extend(self.brakes_train[indices])
                batch_speeds.extend(self.speeds_train[indices])
                # assumes all h5 files has exactly 200 samples (which is verified already)
                batch_imgs.extend([self.imgs_train[int(i / 200)]['rgb'][i % 200] for i in indices])

            # Augment data
            if augment_data:
                idx_to_augment = np.random.randint(low=0, high=len(batch_imgs), size=int(len(batch_imgs) * self.augmentation_ratio))
                augmented_images = self.seq_augmentation.augment_images(np.array([batch_imgs[i] for i in idx_to_augment],
                                                                            dtype=np.uint8))
                i = 0
                for id in idx_to_augment:
                    batch_imgs[id] = augmented_images[i]
                    i += 1

            # batch_high_lvl_cmds: 0 Right, 1 Straight, 2 Follow lane, 3 Left
            yield (np.array(batch_imgs)/255., np.array(batch_imgs)/255., np.array(batch_imgs)/255.,
                   np.array(batch_imgs)[:,:,:,0]/255., np.array(batch_steers), np.array(batch_throttles), np.array(batch_brakes),
                   np.array(batch_speeds), np.array(batch_high_lvl_cmds), np.array(data_indices)), num_of_batches

    # Provides images are from 0 to 1, this should be respected during inference time
    # Minibatches contain an equal number of samples with each command
    def iterate_batches_IL_dataset_val(self, shuffle_data=True, augment_data=False, batch_size=-1, sample_percent=100):
        if batch_size == -1:
            batch_size = self.batch_size

        # Shuffle validation data
        shuffled_idx = np.arange(len(self.steers_val))
        if shuffle_data:
            np.random.shuffle(shuffled_idx)
        shuffled_idx = shuffled_idx[0:int(len(shuffled_idx)*sample_percent/100.)]  # Sample from data

        # Split data by High Level Command
        bins_idx = [[] for _ in range(4)]
        for cmd in range(4):
            bins_idx[cmd] = [i for i in shuffled_idx if (self.high_lvl_cmds_val[i] == cmd)]

        # Iterate the epoch batches
        epoch_size_min_bin = min([len(bins_idx[i]) for i in range(len(bins_idx))])
        stride = int(batch_size / 4)  # batch_size should be multiple of 4 commands
        num_of_batches = len(range(0, epoch_size_min_bin, stride))
        for batch_start_id in range(0, epoch_size_min_bin, stride):
            to = min(epoch_size_min_bin, batch_start_id + stride)

            data_indices = []  # used to return it for just for debugging issues
            batch_high_lvl_cmds = []
            batch_steers = []
            batch_throttles = []
            batch_brakes = []
            batch_speeds = []
            batch_imgs = []
            for cmd in range(4):
                indices = bins_idx[cmd][batch_start_id:to]

                data_indices.extend(indices)
                batch_high_lvl_cmds.extend(self.high_lvl_cmds_val[indices])
                batch_steers.extend(self.steers_val[indices])
                batch_throttles.extend(self.throttles_val[indices])
                batch_brakes.extend(self.brakes_val[indices])
                batch_speeds.extend(self.speeds_val[indices])
                # assumes all h5 files has exactly 200 samples (which is verified already)
                batch_imgs.extend([self.imgs_val[int(i / 200)]['rgb'][i % 200] for i in indices])

            # batch_high_lvl_cmds: 0 Right, 1 Straight, 2 Follow lane, 3 Left
            yield (np.array(batch_imgs)/255., np.array(batch_imgs)/255., np.array(batch_imgs)/255.,
                   np.array(batch_imgs)[:,:,:,0]/255., np.array(batch_steers), np.array(batch_throttles), np.array(batch_brakes),
                   np.array(batch_speeds), np.array(batch_high_lvl_cmds), np.array(data_indices)), num_of_batches

    """# Single Branch each Minibatch iterators:
    def iterate_batches_auc1_dataset_train_singlebranch(self, shuffle_data=True, augment_data=False, batch_size=-1,
                                          sample_percent=100):
        if batch_size == -1:
            batch_size = self.batch_size

        # Shuffle training data
        shuffled_idx = np.arange(len(self.steers_train))
        if shuffle_data:
            np.random.shuffle(shuffled_idx)
        shuffled_idx = shuffled_idx[0:int(len(shuffled_idx) * sample_percent / 100.)]  # Sample from data

        # Split data by High Level Command
        bins_idx = [[] for _ in range(4)]
        for cmd in range(4):
            bins_idx[cmd] = [i for i in shuffled_idx if (self.high_lvl_cmds_train[i] == cmd)]

        # Iterate the epoch batches
        stride = batch_size  # stride might help if we used LSTM
        epoch_size_min_bin = min([len(bins_idx[i]) for i in range(len(bins_idx))])
        num_of_batches = len(range(0, epoch_size_min_bin, stride)) * 4  # x 4 commands

        if augment_data:
            seq_augmentation = self.create_augmenter()

        # Select a random branch (high level command or bin) and take a batch from it
        for batch_start_id in range(0, epoch_size_min_bin, stride):
            for cmd in range(4):
                to = min(epoch_size_min_bin, batch_start_id + stride)
                indices = bins_idx[cmd][batch_start_id:to]

                batch_high_lvl_cmds = self.high_lvl_cmds_train[indices] - 1
                batch_steers = self.steers_train[indices]
                batch_throttles = self.throttles_train[indices]
                batch_brakes = self.brakes_train[indices]
                batch_speeds = self.speeds_train[indices]
                batch_images_cam_1 = [cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB) for img in
                                      self.cam1_image_files_train[indices]]
                batch_images_cam_2 = [cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB) for img in
                                      self.cam2_image_files_train[indices]]
                batch_images_cam_3 = [cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB) for img in
                                      self.cam3_image_files_train[indices]]
                batch_images_pgm = [cv2.imread(img.replace('.png', '.jpg'), cv2.IMREAD_GRAYSCALE) for img in
                                    self.pgm_image_files_train[indices]]

                # TODO: augment data
                # if augment_data:

                # batch_high_lvl_cmds: 0 Follow lane, 1 Right, 2 Left, 3 Straight
                yield (batch_images_cam_1/255., batch_images_cam_2/255., batch_images_cam_3/255., batch_images_pgm/255.,
                       batch_steers, batch_throttles, batch_brakes, batch_speeds, batch_high_lvl_cmds), num_of_batches

    def iterate_batches_auc1_dataset_val_singlebranch(self, shuffle_data=True, augment_data=False, batch_size=-1,
                                        sample_percent=100):
        if batch_size == -1:
            batch_size = self.batch_size

        # Shuffle validation data
        shuffled_idx = np.arange(len(self.steers_val))
        if shuffle_data:
            np.random.shuffle(shuffled_idx)
        shuffled_idx = shuffled_idx[0:int(len(shuffled_idx) * sample_percent / 100.)]  # Sample from data

        # Split data by High Level Command
        bins_idx = [[] for _ in range(4)]
        for cmd in range(4):
            bins_idx[cmd] = [i for i in shuffled_idx if (self.high_lvl_cmds_val[i] == cmd)]

        # Iterate the epoch batches
        stride = batch_size  # stride might help if we used LSTM
        epoch_size_min_bin = min([len(bins_idx[i]) for i in range(len(bins_idx))])
        num_of_batches = len(range(0, epoch_size_min_bin, stride)) * 4  # x 4 commands

        # Select a random branch (high level command or bin) and take a batch from it
        for batch_start_id in range(0, epoch_size_min_bin, stride):
            for cmd in range(4):
                to = min(epoch_size_min_bin, batch_start_id + stride)
                indices = bins_idx[cmd][batch_start_id:to]

                batch_high_lvl_cmds = self.high_lvl_cmds_val[indices] - 1
                batch_steers = self.steers_val[indices]
                batch_throttles = self.throttles_val[indices]
                batch_brakes = self.brakes_val[indices]
                batch_speeds = self.speeds_val[indices]
                batch_images_cam_1 = [cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB) for img in
                                      self.cam1_image_files_val[indices]]
                batch_images_cam_2 = [cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB) for img in
                                      self.cam2_image_files_val[indices]]
                batch_images_cam_3 = [cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB) for img in
                                      self.cam3_image_files_val[indices]]
                batch_images_pgm = [cv2.imread(img.replace('.png', '.jpg'), cv2.IMREAD_GRAYSCALE) for img in
                                    self.pgm_image_files_val[indices]]

                # TODO: augment data
                # if augment_data:

                # batch_high_lvl_cmds: 0 Follow lane, 1 Right, 2 Left, 3 Straight
                yield (batch_images_cam_1/255., batch_images_cam_2/255., batch_images_cam_3/255., batch_images_pgm/255.,
                       batch_steers, batch_throttles, batch_brakes, batch_speeds, batch_high_lvl_cmds), num_of_batches

    def iterate_batches_IL_dataset_train_singlebranch(self, shuffle_data=True, augment_data=False, batch_size=-1,
                                         sample_percent=100):
        if batch_size == -1:
            batch_size = self.batch_size

        # Shuffle training data
        shuffled_idx = np.arange(len(self.steers_train))
        if shuffle_data:
            np.random.shuffle(shuffled_idx)
        shuffled_idx = shuffled_idx[0:int(len(shuffled_idx) * sample_percent / 100.)]  # Sample from data

        # Split data by High Level Command
        bins_idx = [[] for _ in range(4)]
        for cmd in range(4):
            bins_idx[cmd] = [i for i in shuffled_idx if (self.high_lvl_cmds_train[i] == cmd)]

        # Iterate the epoch batches
        stride = batch_size  # stride might help if we used LSTM
        epoch_size_min_bin = min([len(bins_idx[i]) for i in range(len(bins_idx))])
        num_of_batches = len(range(0, epoch_size_min_bin, stride)) * 4  # x 4 commands

        # Select a random branch (high level command or bin) and take a batch from it
        for batch_start_id in range(0, epoch_size_min_bin, stride):
            for cmd in range(4):
                to = min(epoch_size_min_bin, batch_start_id + stride)
                indices = bins_idx[cmd][batch_start_id:to]

                batch_high_lvl_cmds = self.high_lvl_cmds_train[indices]
                batch_steers = self.steers_train[indices]
                batch_throttles = self.throttles_train[indices]
                batch_brakes = self.brakes_train[indices]
                batch_speeds = self.speeds_train[indices]
                # assumes all h5 files has exactly 200 samples (which is verified already)
                batch_imgs = [self.imgs_train[int(i / 200)]['rgb'][i % 200] / 255. for i in indices]

                # TODO: augment data
                # if augment_data:

                # batch_high_lvl_cmds: 0 Right, 1 Straight, 2 Follow lane, 3 Left
                yield (batch_imgs/255., batch_imgs/255., batch_imgs/255., batch_imgs/255., batch_steers, batch_throttles, batch_brakes, \
                       batch_speeds, batch_high_lvl_cmds), num_of_batches

    def iterate_batches_IL_dataset_val_singlebranch(self, shuffle_data=True, augment_data=False, batch_size=-1, sample_percent=100):
        if batch_size == -1:
            batch_size = self.batch_size

        # Shuffle validation data
        shuffled_idx = np.arange(len(self.steers_val))
        if shuffle_data:
            np.random.shuffle(shuffled_idx)
        shuffled_idx = shuffled_idx[0:int(len(shuffled_idx) * sample_percent / 100.)]  # Sample from data

        # Split data by High Level Command
        bins_idx = [[] for _ in range(4)]
        for cmd in range(4):
            bins_idx[cmd] = [i for i in shuffled_idx if (self.high_lvl_cmds_val[i] == cmd)]

        # Iterate the epoch batches
        stride = batch_size  # stride might help if we used LSTM
        epoch_size_min_bin = min([len(bins_idx[i]) for i in range(len(bins_idx))])
        num_of_batches = len(range(0, epoch_size_min_bin, stride)) * 4  # x 4 commands

        # Select a random branch (high level command or bin) and take a batch from it
        for batch_start_id in range(0, epoch_size_min_bin, stride):
            for cmd in range(4):
                to = min(epoch_size_min_bin, batch_start_id + stride)
                indices = bins_idx[cmd][batch_start_id:to]

                batch_high_lvl_cmds = self.high_lvl_cmds_val[indices]
                batch_steers = self.steers_val[indices]
                batch_throttles = self.throttles_val[indices]
                batch_brakes = self.brakes_val[indices]
                batch_speeds = self.speeds_val[indices]
                # assumes all h5 files has exactly 200 samples (which is verified already)
                batch_imgs = [self.imgs_val[int(i / 200)]['rgb'][i % 200] for i in indices]

                # TODO: augment data
                # if augment_data:

                # batch_high_lvl_cmds: 0 Right, 1 Straight, 2 Follow lane, 3 Left
                yield (batch_imgs/255., batch_imgs/255., batch_imgs/255., batch_imgs/255., batch_steers, batch_throttles, batch_brakes, \
                       batch_speeds, batch_high_lvl_cmds), num_of_batches"""
