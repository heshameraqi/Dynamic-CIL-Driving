import math
import numpy as np
import os
from ogm import OGM

from carla import image_converter

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.transforms as mtransforms
import matplotlib.cbook as cbook
import matplotlib.patches as patches
import matplotlib as mpl

class OGM_Planner():
    def __init__(self, route_planner):
        # Visualization settings
        self.visualize = True  # TODO: shouldn't take more than ~8 seconds due to CARLA max timeout in driving_benchmark.py
        self.save_figure = False
        self.visualize_ogm_only = False
        self.high_quality_frame = False  # high quality snap for the paper
        self.high_quality_frame_number = 522  # 434 522 864 (on ex == 4 and pos < 9 CoRL benchamrk)
        self.use_real_car_image = False  # In the OGM displayed ego-car

        self.ogm_map = OGM()
        self.fig = None  # to open a single figure visualization
        self.vpc_circle = None  # for vehicle position circle visualization
        self.scatter = None  # for scan points visualization
        self.sensor_dot = None  # for laser sensor location point visualization
        # The following parameters are in meters, predicted steering is from -70 degrees to 70 degrees
        self.car_width = 2.0 * 0.9399999976158142  # from measurements.player_measurements.bounding_box.extent for the default ego car: Mustang
        self.car_length = 2.0 * 2.3399999141693115  # from measurements.player_measurements.bounding_box.extent for the default ego car: Mustang
        # self.car_width = 2.2
        # self.car_length = 4.4
        self.wheel_base = 2.89  # for the default ego car: Mustang
        self.wheel_radius = 0.32  # for the default ego car: Mustang, not needed for our simple bicycle model
        self.poses_local_future_bicycle = -1  # unset
        self.frames_in_future_bicycle = 30  # after which steering will be 0
        self.frames_skip_bicycle = 10  # self.frames_in_future_bicycle should multiple of it
        self.planner_source_pos_img_pixel_map = None  # Top view colored town map
        self.planner_destination_pos_img_pixel_map = None
        self.planner_current_pos_img_pixel_map = None
        self.planner_source_pos_graph_map = None  # Graph map is the route planner grid map
        self.planner_destination_pos_graph_map = None
        self.planner_current_pos_graph_map = None

        self.planner_current_pos_img_pixel_map_dot = None
        self.planner_current_pos_graph_map_dot = None
        self.planner_source_pos_img_pixel_map_dot = None
        self.planner_source_pos_graph_map_dot = None
        self.planner_destination_pos_img_pixel_map_dot = None
        self.planner_destination_pos_graph_map_dot = None

        self.planner_current_pos_GPS = None
        self.ogm_anchor_pts_spacing = 5  # Shoud be multiple of OGM map height and width pixels
        self.ogm_anchor_occupacny_th = 0.65  # a probability from 0 to 1
        self.ogm_anchor_occupied_pts_th_1 = 50000  # number of occupied cells near a road anchor point to be considered a blocked road
        self.ogm_anchor_occupied_pts_th_2 = 30  # for a road cell, number of occupied cells in all corresponding anchor points, to be considered a blocked road
        self.ogm_anchor_pts = None
        self.ogm_anchor_occ_sums = None
        self.ogm_anchor_pts_occupancy = None
        self.ogm_anchor_pts_occupancy_bool = None  # True for occupied anchor points
        self.ogm_anchor_pts_half_spacing = None
        self.anchor_pts_img_pixel_map = None
        self.anchor_pts_img_pixel_map_dots = None
        self.anchor_pts_graph_map = None
        self.anchor_pts_graph_map_free_dots = None
        self.anchor_pts_graph_map_occupied_dots = None
        self.anchor_pts_graph_map_dots_text = []
        self.anchor_pts_ogm_anchor_dots_text = []

        self.planner_astar_walls = None
        self.planner_astar_route = None

        self.commands_chars_dict = {0: 'G', 5: 'S', 4: 'R', 3: 'L', 2: 'F', 1: '-'}

        # For _project_lidar_to_semantic_seg_image function
        # Get intrinsics matrix
        # TODO: the three values should be read autmoatically from the semantic segmentation camera FoV (experiment_corl17.py)
        self.semseg_camera_fov = 100
        self.semseg_image_size_y = 600
        self.semseg_image_size_x = 800
        f = self.semseg_image_size_x / (2 * np.tan(self.semseg_camera_fov * np.pi / float(360)))
        cx = self.semseg_image_size_x / 2
        cy = self.semseg_image_size_y / 2
        self.intrinsic_mat = np.zeros((3, 3))
        self.intrinsic_mat[0, 0] = f
        self.intrinsic_mat[1, 1] = f
        self.intrinsic_mat[0, 2] = cx
        self.intrinsic_mat[1, 2] = cy
        self.intrinsic_mat[2, 2] = 1.0
        rad_deg = np.pi / float(180)
        self.rad_45 = 45 * rad_deg
        self.rad_135 = 135 * rad_deg
        self.rad_180 = np.pi
        self.rad_225 = 225 * rad_deg
        self.rad_315 = 315 * rad_deg
        self.classes_labels = [
            "Unlabeled",
            "Building",
            "Fence",
            "Other",
            "Pedestrian",
            "Pole",
            "Road line",
            "Road",
            "Sidewalk",
            "Vegetation",
            "Vehicles",
            "Wall",
            "Traffic sign",
            "(Outside camera)",
            "(Overhanging)"
        ]
        self.classes_colors = np.array([
            [0, 0, 0],  # None/Unlabeled
            [70, 70, 70],  # Buildings
            [190, 153, 153],  # Fences
            [72, 0, 90],  # Other
            [220, 20, 60],  # Pedestrians
            [153, 153, 153],  # Poles
            [157, 234, 50],  # RoadLines
            [128, 64, 128],  # Roads
            [244, 35, 232],  # Sidewalks
            [107, 142, 35],  # Vegetation
            [0, 0, 255],  # Vehicles
            [102, 102, 156],  # Walls
            [220, 220, 0],  # TrafficSigns
            [10, 10, 10],  # <Outside camera perspective> indexed by [-2]
            [0, 255, 0]  # <Overhanging objects above car> indexed by [-1]
        ])

        self.route_planner = route_planner

        # Generate OGM anchor points (coordinates are x then y)
        rows = int(self.ogm_map.cfg.height / self.ogm_map.cfg.resolution)
        cols = int(self.ogm_map.cfg.width / self.ogm_map.cfg.resolution)
        step = self.ogm_anchor_pts_spacing
        ogm_anchor_pts = [((step / 2) + (x * step), (step / 2) + (y * step)) for x in range(int(cols / step)) for y in range(int(rows / step))]
        ogm_anchor_pts = np.array([*ogm_anchor_pts])
        self.ogm_anchor_pts = np.full((ogm_anchor_pts.shape[0], 3), 0.22)  # Add z as const just to be able to use planner.converter
        self.ogm_anchor_pts[:, :-1] = ogm_anchor_pts
        self.ogm_anchor_pts = self.ogm_anchor_pts.astype(int)
        self.ogm_anchor_pts_half_spacing = int((self.ogm_anchor_pts[1,1] - self.ogm_anchor_pts[0,1])/2)

    # return the nearest road node regardless how far
    def GPS_to_planner_nearest_road_node(self, GPS_pos):
        # use "_city_track._map.convert_to_node" or "_city_track.project_node" to give the neares road (non-wall) cell)
        return self.route_planner._city_track.project_node(GPS_pos, use_max_scale_of=-1)

    # returns (-1,-1) if the GPS isn't road
    def GPS_to_planner_road_node(self, GPS_pos):
        # use "_city_track._map.convert_to_node" or "_city_track.project_node" to give the neares road (non-wall) cell)
        return self.route_planner._city_track.project_node(GPS_pos, use_max_scale_of=0)  # important parameter, 0 means return direct road nodes only, everything else is returned (-1,-1)

    def set_source_destination_from_GPS(self, source_pos, destination_pos):
        planner_current_pos_img_world_GPS = [source_pos.location.x, source_pos.location.y, 0.22]
        self.planner_destination_pos_img_pixel_map = [destination_pos.location.x, destination_pos.location.y, 0.22]

        # Convert GPS to Nodes map coordinates
        self.planner_source_pos_graph_map = self.GPS_to_planner_nearest_road_node(planner_current_pos_img_world_GPS)
        self.planner_destination_pos_graph_map = self.GPS_to_planner_nearest_road_node(self.planner_destination_pos_img_pixel_map)

        # Convert GPS to world colored map pixels coordinates
        self.planner_source_pos_img_pixel_map = self.route_planner._city_track._map.convert_to_pixel(planner_current_pos_img_world_GPS)
        self.planner_destination_pos_img_pixel_map = self.route_planner._city_track._map.convert_to_pixel(self.planner_destination_pos_img_pixel_map)

        if (self.fig is not None) and (not self.visualize_ogm_only):
            self.planner_current_pos_img_pixel_map_dot.remove()
            self.planner_current_pos_graph_map_dot.remove()
            self.planner_source_pos_img_pixel_map_dot.remove()
            self.planner_source_pos_graph_map_dot.remove()
            self.planner_destination_pos_img_pixel_map_dot.remove()
            self.planner_destination_pos_graph_map_dot.remove()
            self.planner_source_pos_img_pixel_map_dot = self.ax[0, 2].scatter(self.planner_source_pos_img_pixel_map[0],
                                                                              self.planner_source_pos_img_pixel_map[1],
                                                                              c='r', marker='o', s=20, linewidths=2,
                                                                              zorder=2)
            self.planner_destination_pos_img_pixel_map_dot = self.ax[0, 2].scatter(
                self.planner_destination_pos_img_pixel_map[0], self.planner_destination_pos_img_pixel_map[1],
                c='g', marker='o', s=20, linewidths=2, zorder=2)
            self.planner_current_pos_img_pixel_map_dot = self.ax[0, 2].scatter(self.planner_current_pos_img_pixel_map[0],
                                                                               self.planner_current_pos_img_pixel_map[1],
                                                                               c='b', marker='o', s=20, linewidths=2,
                                                                               zorder=2)
            # Note: x and y axis are exchnaged
            self.planner_source_pos_graph_map_dot = self.ax[1, 2].scatter(
                self.planner_source_pos_graph_map[0] + 0.5, self.planner_source_pos_graph_map[1] + 0.5,
                c='r', marker='o', s=20, linewidths=2, zorder=2)
            self.planner_destination_pos_graph_map_dot = self.ax[1, 2].scatter(
                self.planner_destination_pos_graph_map[0] + 0.5, self.planner_destination_pos_graph_map[1] + 0.5,
                c='g', marker='o', s=20, linewidths=2, zorder=2)
            self.planner_current_pos_graph_map_dot = self.ax[1, 2].scatter(self.planner_current_pos_graph_map[0] + 0.5,
                                                                           self.planner_current_pos_graph_map[1] + 0.5,
                                                                           c='b', marker='o', s=20, linewidths=2,
                                                                           zorder=2)

    # current_pos_transform_GPS has the GPS position of the car
    # control is the predicted control from NN
    # control.steer is from -1 to 1, the maximum for steering of the used ego car (Mustang) is 70 degrees
    # planner_control_input: LANE_FOLLOW = 2.0, REACH_GOAL = 0.0, TURN_LEFT = 3.0, TURN_RIGHT = 4.0, GO_STRAIGHT = 5.0
    # returns control as is, or modify it if needed
    def step(self, sensor_data, measurements, control, planner_control_input, current_pos_transform_GPS):
        # Get some measurements
        self.planner_current_pos_GPS = [current_pos_transform_GPS.location.x, current_pos_transform_GPS.location.y, 0.22]

        self.planner_current_pos_graph_map = self.GPS_to_planner_nearest_road_node(self.planner_current_pos_GPS)  # Convert GPS to Nodes map coordinates
        self.planner_current_pos_img_pixel_map = self.route_planner._city_track._map.convert_to_pixel(self.planner_current_pos_GPS)  # Convert GPS to world colored map pixels coordinates

        actual_speed = measurements.player_measurements.forward_speed
        x_abs = measurements.player_measurements.transform.location.x
        y_abs = measurements.player_measurements.transform.location.y
        yaw_abs = measurements.player_measurements.transform.rotation.yaw  # CARLA gives it in degrees
        frame_number = measurements.frame_number

        # Get full scan
        # x = self.lidar_data[:, 0], y = self.lidar_data[:, 1], z = -self.lidar_data[:, 2]
        _lidar_measurement = sensor_data.get('Lidar_ogm_long_range', None)
        lidar_data = np.array(_lidar_measurement.data)
        x = lidar_data[:, 0]
        y = lidar_data[:, 1]
        z = -lidar_data[:, 2]
        fullscan = np.array(list(zip(x, y, z)))

        # Semantic Segmentation-based pointcloud for filtration
        img = image_converter.to_rgb_array(sensor_data.get('CameraRGB_centre', None))
        img_semseg = image_converter.labels_to_cityscapes_palette(sensor_data.get('CameraSemSeg', None))
        fullscan_2d, front_data_idx = self._project_lidar_to_semantic_seg_image(fullscan)
        points_semseg_color = []
        to_filter_out_idx = []
        dynamic_objects_pts_idx = []
        for i in range(len(fullscan_2d[0, :])):
            if i not in front_data_idx:  # Outside camera perspective
                points_semseg_color.append(self.classes_colors[-2])
                to_filter_out_idx.append(i)  # filter scan points outside camera perspective as well
            elif 0 <= fullscan_2d[0, i] < self.semseg_image_size_x and \
                    0 <= fullscan_2d[1, i] < self.semseg_image_size_y:
                points_semseg_color.append(img_semseg[int(fullscan_2d[1, i]), int(fullscan_2d[0, i])])
                # If Road or Road Line, filter them out
                if (img_semseg[int(fullscan_2d[1, i]), int(fullscan_2d[0, i])] == [128, 64, 128]).all() or \
                        (img_semseg[int(fullscan_2d[1, i]), int(fullscan_2d[0, i])] == [157, 234, 50]).all():
                    to_filter_out_idx.append(i)
                # If Vehicle or Pedestrian, filter them out and add them statically to OGM
                elif (img_semseg[int(fullscan_2d[1, i]), int(fullscan_2d[0, i])] == [220, 20, 60]).all() or \
                        (img_semseg[int(fullscan_2d[1, i]), int(fullscan_2d[0, i])] == [0, 0, 255]).all():
                    to_filter_out_idx.append(i)
                    dynamic_objects_pts_idx.append(i)
            else:
                points_semseg_color.append([255, 255, 255])  # Should't happen
        dynamic_objects_scans = fullscan[dynamic_objects_pts_idx].copy()
        fullscan_filtered = fullscan.copy()
        fullscan_filtered = np.delete(fullscan_filtered, to_filter_out_idx, axis=0)

        # Remove scanpoints above car height (overhanging objects): long trees, billboards
        height = 1  # meters/10?
        points_semseg_color = np.array(points_semseg_color)
        points_semseg_color[fullscan[:, 2] > height] = self.classes_colors[-1]
        fullscan_filtered = fullscan_filtered[fullscan_filtered[:, 2] <= height]

        # Downsample scanpoints (systematic: keep each nth point)
        # step = 5
        # fullscan_filtered = fullscan_filtered[np.arange(0, len(fullscan_filtered), step)]

        # Update OGM
        self.ogm_map.draw_ogm_map(x_abs, y_abs, yaw_abs, actual_speed, fullscan_filtered, dynamic_objects_scans)

        # Convert planning_centre_points GPS to world colored map pixels coordinates
        # Coordinates available are (represented as x then y): self.ogm_map.pose_local,
        # self.planner_current_pos_GPS, self.planner_current_pos_img_pixel, self.planner_current_pos_graph_map
        x_diff = self.planner_current_pos_GPS[0] - self.ogm_map.pose_local[0] * self.ogm_map.cfg.resolution
        y_diff = self.planner_current_pos_GPS[1] - self.ogm_map.pose_local[1] * self.ogm_map.cfg.resolution
        self.anchor_pts_GPS = np.zeros(self.ogm_anchor_pts.shape)
        self.anchor_pts_GPS[:, 0] = x_diff + self.ogm_anchor_pts[:, 0] * self.ogm_map.cfg.resolution
        self.anchor_pts_GPS[:, 1] = y_diff + self.ogm_anchor_pts[:, 1] * self.ogm_map.cfg.resolution
        # Convert GPS to world colored map pixels coordinates
        self.anchor_pts_img_pixel_map = np.apply_along_axis(self.route_planner._city_track._map.convert_to_pixel, 1,
                                                            self.anchor_pts_GPS)
        # Convert GPS to Nodes map coordinates
        # All far from road will be mapped to (-1,-1), threshold is in planner.city_track project_node function
        # TODO: this tack of adding shift constants is to handle a bug in converting GPS to graph nodes
        GPS = self.anchor_pts_GPS
        GPS[:, 0] = GPS[:, 0] + 5
        GPS[:, 1] = GPS[:, 1] + 3
        self.anchor_pts_graph_map = np.apply_along_axis(self.GPS_to_planner_road_node, 1, GPS)  # returns repeasted values and (-1,-1) for non-road GPS

        # Update occupancy for each OGM anchor point
        self.ogm_anchor_occ_sums = np.zeros((self.anchor_pts_graph_map.shape[0]))
        self.ogm_anchor_pts_occupancy = np.zeros((self.anchor_pts_graph_map.shape[0]))  # array of Falses
        self.ogm_anchor_pts_occupancy_bool = np.zeros((self.anchor_pts_graph_map.shape[0]), dtype=bool)  # array of Falses
        for i, p in enumerate(self.ogm_anchor_pts):
            ogm_slice = self.ogm_map.map_without_dynamic_prob[p[1]-self.ogm_anchor_pts_half_spacing : p[1]+self.ogm_anchor_pts_half_spacing,
                        p[0]-self.ogm_anchor_pts_half_spacing : p[0]+self.ogm_anchor_pts_half_spacing]
            self.ogm_anchor_occ_sums[i] = len(ogm_slice[ogm_slice > self.ogm_anchor_occupacny_th])
            if self.anchor_pts_graph_map[i,0] == -1:
                self.ogm_anchor_occ_sums[i] = -1
            elif self.ogm_anchor_occ_sums[i] > self.ogm_anchor_occupied_pts_th_1:
                self.ogm_anchor_pts_occupancy_bool[i] = True
        # delete (-1,-1) rows from anchor points and their associated values
        self.ogm_anchor_pts_occupancy = self.ogm_anchor_pts_occupancy[self.anchor_pts_graph_map.min(axis=1) >= 0]
        self.ogm_anchor_pts_occupancy_bool = self.ogm_anchor_pts_occupancy_bool[self.anchor_pts_graph_map.min(axis=1) >= 0]
        self.ogm_anchor_occ_sums = self.ogm_anchor_occ_sums[self.anchor_pts_graph_map.min(axis=1) >= 0]
        self.anchor_pts_graph_map = self.anchor_pts_graph_map[self.anchor_pts_graph_map.min(axis=1) >= 0, :]
        # Select unique graph cells
        self.anchor_pts_graph_map, inverse_indices = np.unique(self.anchor_pts_graph_map, axis=0, return_inverse=True)
        uniaue_bools = np.zeros((self.anchor_pts_graph_map.shape[0]), dtype=bool)  # array of Falses
        unique_occupancies = np.zeros((self.anchor_pts_graph_map.shape[0]))
        for i in range(self.anchor_pts_graph_map.shape[0]):
            uniaue_bools[i] = np.any(self.ogm_anchor_pts_occupancy_bool[inverse_indices==i])
            unique_occupancies[i] = np.sum(self.ogm_anchor_occ_sums[inverse_indices==i])
            if unique_occupancies[i] > self.ogm_anchor_occupied_pts_th_2:
                uniaue_bools[i] = True
        self.ogm_anchor_pts_occupancy_bool = uniaue_bools
        self.ogm_anchor_pts_occupancy = unique_occupancies

        # Add walls to occupied planner grid cells
        # for i, v in enumerate(self.ogm_anchor_pts_occupancy_bool):
        #     if v:

        # Predict future pose using bicycle model
        # self.ogm_map.pose_local[2] = math.radians(70)  # self.ogm_map.pose_local: [cells], [cells], [radians counter clockwise]
        # steering_angle = 0.3  # -1:1 (-70 to 70 degrees), 1 means 70 degrees to the car right
        # actual_speed = 20  # m/s
        # self.frames_in_future_bicycle = 15
        self.poses_local_future_bicycle = self._bicycle_model(self.ogm_map.pose_local, control.steer, actual_speed,
                                                              self.frames_in_future_bicycle, self.frames_skip_bicycle)

        # Visualize Map
        self._visualize_planning(frame_number, actual_speed, control, planner_control_input, fullscan,
                                 points_semseg_color, img, img_semseg)

        return control

    def _project_lidar_to_semantic_seg_image(self, fullscan):
        # Get front-facing points
        front_data_idx = []
        right_data_idx = []
        rear_data_idx = []
        left_data_idx = []
        for i, p in enumerate(fullscan):
            rad = np.arctan2(-p[1], -p[0])
            if self.rad_45 <= rad < self.rad_135:
                front_data_idx.append(i)
            elif self.rad_135 <= rad < self.rad_180 or -self.rad_180 <= rad < -self.rad_135:
                right_data_idx.append(i)
            elif -self.rad_135 <= rad < -self.rad_45:
                rear_data_idx.append(i)
            elif -self.rad_45 <= rad < self.rad_45:
                left_data_idx.append(i)

        # Project lidar to image
        scans_matrix = np.transpose(fullscan).copy()
        scans_matrix[2, :] = -scans_matrix[2, :]

        scans_matrix[:, front_data_idx] = scans_matrix[[0, 2, 1], :][:, front_data_idx]
        scans_matrix[:, rear_data_idx] = scans_matrix[[0, 2, 1], :][:, rear_data_idx]
        scans_matrix[2, front_data_idx] = -scans_matrix[2, front_data_idx]
        scans_matrix[0, rear_data_idx] = -scans_matrix[0, rear_data_idx]

        scans_matrix[:, left_data_idx] = scans_matrix[[1, 2, 0], :][:, left_data_idx]
        scans_matrix[:, right_data_idx] = scans_matrix[[1, 2, 0], :][:, right_data_idx]
        scans_matrix[0, left_data_idx] = -scans_matrix[0, left_data_idx]
        scans_matrix[2, left_data_idx] = -scans_matrix[2, left_data_idx]

        fullscan_2d = self.intrinsic_mat @ scans_matrix
        class_colors = []
        for i in range(len(fullscan_2d[0, :])):
            fullscan_2d[:, i] = fullscan_2d[:, i] / float(fullscan_2d[2, i])
            if fullscan_2d[0, i] - np.floor(fullscan_2d[0, i]) > 0.5:
                fullscan_2d[0, i] = np.ceil(fullscan_2d[0, i])
            else:
                fullscan_2d[0, i] = np.floor(fullscan_2d[0, i])
            if fullscan_2d[1, i] - np.floor(fullscan_2d[1, i]) > 0.5:
                fullscan_2d[1, i] = np.ceil(fullscan_2d[1, i])
            else:
                fullscan_2d[1, i] = np.floor(fullscan_2d[1, i])

        """
        0: [0, 0, 0],         # None/Unlabeled
        1: [70, 70, 70],      # Buildings
        2: [190, 153, 153],   # Fences
        3: [72, 0, 90],       # Other
        4: [220, 20, 60],     # Pedestrians
        5: [153, 153, 153],   # Poles
        6: [157, 234, 50],    # RoadLines
        7: [128, 64, 128],    # Roads
        8: [244, 35, 232],    # Sidewalks
        9: [107, 142, 35],    # Vegetation
        10: [0, 0, 255],      # Vehicles
        11: [102, 102, 156],  # Walls
        12: [220, 220, 0]     # TrafficSigns
        """
        return fullscan_2d, front_data_idx

    # Assumes that speed will decay to zero in frames_in_future_bicycle
    # pose_local = [x,y of vehicle centre point in map cells (not meters), yaw in radians counter-clockwise]
    # steering_angle: tire steering angle is from -1:1 (-70 to 70 degrees), 1 means 70 degrees to the car right (extreme angle #TODO: as in model training data)
    # actual_speed is in m/s
    # steering_angle is from -1 (extrmete left) to 1 (extrmete right), the maximum for steering of the used ego car (Mustang) is 70 degrees
    def _bicycle_model(self, pose_local, steering_angle, actual_speed, frames_in_future, frames_to_skip):
        steering_angle = -steering_angle * math.radians(70)
        seconds_dt = frames_to_skip / 15.0  # # FPS is experiment_corl17.py is 15, TODO: this should be handled automatically

        steps = len(range(0, frames_in_future, frames_to_skip))  # +1 to be inclusive
        poses_local_future_bicycle = np.zeros((steps, 3))
        current_pose = np.array(pose_local)
        actual_speed = actual_speed / self.ogm_map.cfg.resolution
        start_steering_angle = steering_angle

        for i in range(steps):
            pose_local_future = np.zeros(3)
            frontWheel = current_pose[0:2] + self.wheel_base / 2. * np.array([math.cos(current_pose[2]),
                                                                              -math.sin(current_pose[2])])
            backWheel = current_pose[0:2] - self.wheel_base / 2. * np.array([math.cos(current_pose[2]),
                                                                             -math.sin(current_pose[2])])
            backWheel = backWheel + actual_speed * seconds_dt * np.array([math.cos(current_pose[2]),
                                                                          -math.sin(current_pose[2])])
            frontWheel = frontWheel + actual_speed * seconds_dt * np.array([math.cos(current_pose[2] + steering_angle),
                                                                            -math.sin(
                                                                                current_pose[2] + steering_angle)])
            pose_local_future[0:2] = (frontWheel + backWheel) / 2
            pose_local_future[2] = math.atan2(-(frontWheel[1] - backWheel[1]), frontWheel[0] - backWheel[0])

            current_pose = np.array(pose_local_future)
            poses_local_future_bicycle[i] = pose_local_future
            steering_angle -= start_steering_angle / (steps - 1)
        return poses_local_future_bicycle

    def _visualize_planning(self, frame_number, actual_speed, control,
                            planner_control_input, fullscan, points_semseg_color, img, img_semseg):
        if (not self.high_quality_frame and (self.visualize or self.save_figure)) or (
                self.high_quality_frame and self.save_figure and frame_number == self.high_quality_frame_number):
            if not self.fig:  # First time figure is created
                if self.visualize_ogm_only:
                    self.fig = plt.figure(figsize=(8, 8))
                    self.ax_ogm = plt.gca()
                else:
                    self.fig, self.ax = plt.subplots(2, 3, figsize=(16, 8))
                    self.ax_ogm = self.ax[0, 1]
                    self.ax_ogm.set_xlabel('Meters')
                    self.ax_ogm.set_ylabel('Meters')
                    plt.subplots_adjust(right=0.3)
                    self.ax[0, 0].set_title('Centre Camera')
                    self.ax[1, 0].set_title('Semantic Segmentation')
                    self.ax[0, 2].set_title('World Map')
                    self.ax[1, 2].set_title('Route Planner Map (one way)')
                    ax = self.ax[1, 1]
                    ax.set_title('LiDAR Poincloud')
                    ax.set_aspect('equal')
                    # ax.set_yticklabels([])
                    # ax.set_xticklabels([])
                    ax.set_xlim(-50, 50)  # equal to +- LiDAR range
                    ax.set_ylim(-20, 100)  # equal to +- LiDAR range
                    ax.set_xlabel('Meters')
                    ax.set_ylabel('Meters')

                    # Legends
                    markers = [Line2D([0], [0], marker='s', color='w', label='Scatter', markerfacecolor=c,
                                               markersize=12, markeredgecolor=[0., 0., 0.]) for c in
                                        [[0., 1., 0.], [1., 0., 0.], [1., 1., 1.], [0., 0., 0.]]]
                    markers.extend([Line2D([], [], marker='.', color='w', markeredgecolor=c, label='Scatter',
                                           markerfacecolor='w', markersize=5) for c in [[1., 1., 0.]]])
                    self.ax_ogm.legend(markers, ['Ego-vehicle', 'Motion Model', 'Occupied', 'Free', 'Anchor Points'],
                                       loc='center left',
                                       bbox_to_anchor=(1.01, 0.5))
                    self.ax[1, 1].legend([Line2D([0], [0], marker='o', color='w', label='Scatter', markerfacecolor=c,
                                                 markersize=10) for c in self.classes_colors / 255.],
                                         self.classes_labels, loc='center left', bbox_to_anchor=(1.01, 0.5))
                    markers = [Line2D([0], [0], marker='o', color='w', label='Scatter', markerfacecolor=c,
                                                 markersize=10) for c in [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]]
                    markers.extend([Line2D([], [], marker='.', color='w', markeredgecolor=c, label='Scatter',
                                           markerfacecolor='w', markersize=5) for c in [[1., 1., 0.]]])
                    self.ax[0, 2].legend(markers,
                                         ['Source', 'Destination', 'Current', 'OGM Anchor\nPoints'], loc='center left',
                                         bbox_to_anchor=(1.01, 0.5))
                    markers = [Line2D([0], [0], marker='o', color='w', label='Scatter', markerfacecolor=c,
                                      markersize=10) for c in [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]]
                    markers.extend([Line2D([], [], marker='s', color='w', markeredgecolor=[0., 0., 0.], label='Scatter',
                                           markerfacecolor='#D3D3D3', markersize=5)])
                    markers.extend([Line2D([], [], marker='x', color='w', markeredgecolor=c, label='Scatter',
                                           markerfacecolor='w', markersize=5) for c in [[0., 0., 0.]]])
                    markers.extend([Line2D([], [], marker='.', color='w', markeredgecolor=c, label='Scatter',
                                           markerfacecolor=c, markersize=5) for c in [[1., 0., 0.]]])
                    markers.extend([Line2D([], [], marker='o', color='w', markeredgecolor=c, label='Scatter',
                                           markerfacecolor='w', markersize=10) for c in [[1., 1., 0.], [0., 0., 0.]]])
                    self.ax[1, 2].legend(markers, ['Source', 'Destination', 'Current', 'Off-road', 'Walls', 'A* Route',
                                                   'OGM Free', 'OGM Occupied'], loc='center left',
                                         bbox_to_anchor=(1.01, 0.5))

                    # Display camera and semseg images
                    self.img_obj = self.ax[0, 0].imshow(img)
                    self.ax[0, 0].axis('off')
                    self.img_semseg_obj = self.ax[1, 0].imshow(img_semseg.astype(int))
                    self.ax[1, 0].axis('off')

                    # Display route planner world colored map
                    self.planner_world_map = self.ax[0, 2].imshow(self.route_planner._city_track._map.map_image,
                                                                  origin='upper')
                    self.planner_source_pos_img_pixel_map_dot = self.ax[0, 2].scatter(self.planner_source_pos_img_pixel_map[0],
                                                                                      self.planner_source_pos_img_pixel_map[1],
                                                                                      c='r', marker='o', s=20, linewidths=2)
                    self.planner_destination_pos_img_pixel_map_dot = self.ax[0, 2].scatter(
                        self.planner_destination_pos_img_pixel_map[0], self.planner_destination_pos_img_pixel_map[1],
                        c='g', marker='o', s=20, linewidths=2)
                    self.planner_current_pos_img_pixel_map_dot = self.ax[0, 2].scatter(self.planner_current_pos_img_pixel_map[0],
                                                                                       self.planner_current_pos_img_pixel_map[1],
                                                                                       c='b', marker='o', s=20, linewidths=2)
                    self.anchor_pts_img_pixel_map_dots = self.ax[0, 2].scatter(self.anchor_pts_img_pixel_map[:, 0],
                                                                           self.anchor_pts_img_pixel_map[:, 1],
                                                                               c='y', marker='.', s=10, linewidths=1)
                    self.ax[0, 2].set_yticks([])
                    self.ax[0, 2].set_xticks([])

                    # Display route planner graph map, show grid lines to indicate quantization of the nodes grid map (Note: x and y axis are exchanged)
                    from matplotlib import colors as mplcolors
                    cMap = mplcolors.ListedColormap(['w', '#D3D3D3'])  # '#D3D3D3' is light gray
                    self.ax[1, 2].pcolormesh(self.route_planner._city_track._map._grid._structure.T, edgecolors='k',
                                             linewidth=0.05 ,cmap=cMap, antialiased=True)
                    # OGM anchors in planner grid map
                    if not all(self.ogm_anchor_pts_occupancy_bool):
                        self.anchor_pts_graph_map_free_dots = self.ax[1, 2].scatter(
                            self.anchor_pts_graph_map[~self.ogm_anchor_pts_occupancy_bool][:, 0] + 0.5,
                            self.anchor_pts_graph_map[~self.ogm_anchor_pts_occupancy_bool][:, 1] + 0.5,
                            facecolors='none', edgecolors='y', marker='o', s=40, linewidths=1.0, zorder=3)
                        # Annotattion of occupancy values
                        '''for i, txt in enumerate(self.ogm_anchor_pts_occupancy[~self.ogm_anchor_pts_occupancy_bool]):
                            ann = self.ax[1, 2].annotate(
                                int(txt), (self.anchor_pts_graph_map[~self.ogm_anchor_pts_occupancy_bool][i, 0] + 0.5,
                                      self.anchor_pts_graph_map[~self.ogm_anchor_pts_occupancy_bool][i, 1] + 0.5),
                                size=5, color='k', weight='bold')
                            self.anchor_pts_graph_map_dots_text.append(ann)'''
                    if any(self.ogm_anchor_pts_occupancy_bool):
                        self.anchor_pts_graph_map_occupied_dots = self.ax[1, 2].scatter(
                            self.anchor_pts_graph_map[self.ogm_anchor_pts_occupancy_bool][:, 0] + 0.5,
                            self.anchor_pts_graph_map[self.ogm_anchor_pts_occupancy_bool][:, 1] + 0.5,
                            facecolors='none', edgecolors='k', marker='o', s=40, linewidths=1.0, zorder=3)
                        # Annotattion of occupancy values
                        '''for i, txt in enumerate(self.ogm_anchor_pts_occupancy[self.ogm_anchor_pts_occupancy_bool]):
                            ann = self.ax[1, 2].annotate(
                                int(txt), (self.anchor_pts_graph_map[self.ogm_anchor_pts_occupancy_bool][i, 0] + 0.5,
                                      self.anchor_pts_graph_map[self.ogm_anchor_pts_occupancy_bool][i, 1] + 0.5),
                                    size=5, color='k', weight='bold')
                            self.anchor_pts_graph_map_dots_text.append(ann)'''
                    self.ax[1, 2].invert_yaxis()
                    self.ax[1, 2].set_aspect('equal')
                    self.ax[1, 2].set_yticks([])
                    self.ax[1, 2].set_xticks([])
                    self.planner_source_pos_graph_map_dot = self.ax[1, 2].scatter(
                        self.planner_source_pos_graph_map[0] + 0.5, self.planner_source_pos_graph_map[1] + 0.5,
                        c='r', marker='o', s=20, linewidths=2, zorder=2)
                    self.planner_destination_pos_graph_map_dot = self.ax[1, 2].scatter(
                        self.planner_destination_pos_graph_map[0] + 0.5, self.planner_destination_pos_graph_map[1] + 0.5,
                        c='g', marker='o', s=20, linewidths=2, zorder=2)
                    self.planner_current_pos_graph_map_dot = self.ax[1, 2].scatter(self.planner_current_pos_graph_map[0] + 0.5,
                                                                                   self.planner_current_pos_graph_map[1] + 0.5,
                                                                                   c='b', marker='o', s=20, linewidths=2,
                                                                                   zorder=2)

                # Draw OGM
                # binary_map = np.where(map < 0.0005, 1, 0)  # The lower the threshold the more occupation in map
                # self.pgm_img_obj = self.ax_ogm.imshow(binary_map, 'Greys')
                self.ogm_img_obj = self.ax_ogm.imshow(self.ogm_map.map_without_dynamic_prob, cmap='gray', vmin=0, vmax=1)
                labels_positions = np.arange(0, self.ogm_map.map_without_dynamic_prob.shape[0], 40)  # the last number in arange is the step
                labels_new = labels_positions*self.ogm_map.cfg.resolution
                self.ax_ogm.set_xticks(labels_positions)
                self.ax_ogm.set_xticklabels(labels_new.astype(int))
                labels_positions = np.arange(0, self.ogm_map.map_without_dynamic_prob.shape[1], 40)  # the last number in arange is the step
                labels_new = labels_positions * self.ogm_map.cfg.resolution
                self.ax_ogm.set_yticks(labels_positions)
                self.ax_ogm.set_yticklabels(labels_new.astype(int))
                # Draw OGM route planner centers
                self.ax_ogm.scatter(self.ogm_anchor_pts[:,0], self.ogm_anchor_pts[:,1], c='y', marker='.', s=10,
                                    linewidths=0.5)

                # Vehicle position circle
                self.vpc_circle = plt.Circle((self.ogm_map.map_centre_cell[0], self.ogm_map.map_centre_cell[1]), 0,
                                             color='y', fill=False)
                self.vpc_circle.radius = self.ogm_map.voc_radius
                self.ax_ogm.add_artist(self.vpc_circle)
                # Vehicle
                self.car_width_on_map = self.car_width / self.ogm_map.cfg.resolution  # 51
                self.car_length_on_map = self.car_length / self.ogm_map.cfg.resolution  # 25
                if self.use_real_car_image:
                    car_img_arr = plt.imread(cbook.get_sample_data(os.getcwd() + "/assets/car_small.png"))
                    self.car_img = self.ax_ogm.imshow(car_img_arr, interpolation='none',
                                                      extent=[0, self.car_length_on_map, 0, self.car_width_on_map],
                                                      clip_on=True,
                                                      alpha=1.0)  # extent=[0, self.car_width, 0, self.car_length]
                    self._shift_rotate_car_img(self.ax_ogm, self.ogm_map.pose_local)
                    self.ax_ogm.set_xlim(0, self.ogm_map.map_without_dynamic_prob.shape[1])  # equal to +- LiDAR range
                    self.ax_ogm.set_ylim(0, self.ogm_map.map_without_dynamic_prob.shape[0])  # equal to +- LiDAR range
                else:  # Draw a rectangle car
                    # Future cars bicycle model: Blue
                    self.car_rects_future = []
                    car_rects_in_future_count = len(range(0, self.frames_in_future_bicycle, self.frames_skip_bicycle))
                    for i in range(car_rects_in_future_count):
                        self.car_rects_future.append(
                            patches.Rectangle((0, 0), self.car_length_on_map, self.car_width_on_map,
                                              linewidth=1, edgecolor='k', facecolor='r', alpha=0.75))
                        self._shift_rotate_car_rect(self.ax_ogm, self.car_rects_future[-1],
                                                    self.poses_local_future_bicycle[i])
                        self.ax_ogm.add_patch(self.car_rects_future[-1])
                    # Current car: Green
                    self.car_rect = patches.Rectangle((0, 0), self.car_length_on_map, self.car_width_on_map,
                                                      linewidth=1, edgecolor='k', facecolor='g', alpha=1.0)
                    self._shift_rotate_car_rect(self.ax_ogm, self.car_rect, self.ogm_map.pose_local)
                    self.ax_ogm.add_patch(self.car_rect)
                    # TODO: Current previously predicted car: Red

                # Visualize sensor dot
                self.sensor_dot = self.ax_ogm.scatter(self.ogm_map.pose_local[0], self.ogm_map.pose_local[1],
                                                      c='r', marker='.', s=20, linewidths=2)

                # Figure adjustments
                plt.tight_layout()
                plt.subplots_adjust(top=0.9, bottom=0.05, hspace=0.5, wspace=0.2)  # For multiplelines (long) OGM figure title, also horizontal and vertical spacing
                # plt.show(block=False)  # To see the figure (optional)
            else:  # Not first time figure created
                if not self.visualize_ogm_only:
                    # Update images and OGM
                    self.img_obj.set_data(img)
                    self.img_semseg_obj.set_data(img_semseg.astype(int))
                    # Remove old scan points
                    if self.scatter:
                        self.scatter.remove()
                    # Visualize pointcloud
                    """step = 10  # For faster visualization
                    idx = np.arange(0, len(fullscan), step)
                    self.scatter = self.ax[1, 1].scatter(fullscan[idx, 0], -fullscan[idx, 1],
                                                         c=np.array(points_semseg_color[idx]) / 255., marker='.', s=1.0,
                                                         linewidths=0.8)  # alpha=0.7"""
                    self.scatter = self.ax[1, 1].scatter(fullscan[:, 0], -fullscan[:, 1],
                                                         c=np.array(points_semseg_color) / 255., marker='.', s=10.0,
                                                         linewidths=0.8)  # alpha=0.7

                    # Update route planner world colored map
                    self.anchor_pts_img_pixel_map_dots.remove()
                    self.anchor_pts_img_pixel_map_dots = self.ax[0, 2].scatter(self.anchor_pts_img_pixel_map[:, 0],
                                                                           self.anchor_pts_img_pixel_map[:, 1],
                                                                               c='y', marker='.', s=10, linewidths=1)

                    # Update route planner graph map, Note: x and y axis are exchnaged
                    '''self.ax[1, 2].set_title('Route Planner Map (one way roads)\nCurrent Command: %s\nFar from intersection (LANE_FOLLOW): %s\nNext Intersections CMDs: %s' %
                                            (planner_control_input, "Yes" if self.route_planner.far_from_intersection else "No",
                                             str([self.commands_chars_dict[x] for x in self.route_planner._commands])))'''
                    self.ax[1, 2].set_title('Route Planner Map (one way roads)\nCurrent Command: %s' % (planner_control_input))
                    self.planner_current_pos_img_pixel_map_dot.remove()
                    self.planner_current_pos_img_pixel_map_dot = self.ax[0, 2].scatter(self.planner_current_pos_img_pixel_map[0],
                                                                                       self.planner_current_pos_img_pixel_map[1],
                                                                                       c='b', marker='o', s=20, linewidths=2)
                    self.planner_current_pos_graph_map_dot.remove()
                    self.planner_current_pos_graph_map_dot = self.ax[1, 2].scatter(self.planner_current_pos_graph_map[0] + 0.5,
                                                                                   self.planner_current_pos_graph_map[1] + 0.5,
                                                                                   c='b', marker='o', s=20,
                                                                                   linewidths=2, zorder=2)
                    # self.planner_world_map.set_data(self.route_planner._city_track._map.map_image)
                    # self.planner_grid_map.set_data(self.route_planner._city_track._map._grid._structure)
                    if self.planner_astar_walls:
                        self.planner_astar_walls.remove()
                    if self.planner_astar_route:
                        self.planner_astar_route.remove()
                    walls = [x for x in self.route_planner._city_track.astar_cells if not x.reachable]
                    if walls:
                        self.planner_astar_walls = self.ax[1, 2].scatter([i.x+0.5 for i in walls], [i.y+0.5 for i in walls],
                                                                         c='k', marker='x', s=20, linewidths=1, zorder=3)
                    route = np.array([*self.route_planner._city_track.astar_route])  # list of tuples to 2D array
                    if route is not None:
                        self.planner_astar_route = self.ax[1, 2].scatter(route[:, 0]+0.5, route[:, 1]+0.5, c='r',
                                                                         marker='.', s=20, linewidths=1)
                    # OGM anchors in planner grid map
                    if self.anchor_pts_graph_map_free_dots is not None:
                        try:  # TODO: remove try and fix that unknown Error
                            self.anchor_pts_graph_map_free_dots.remove()
                        except:
                            pass
                    if self.anchor_pts_graph_map_occupied_dots is not None:
                        try:  # TODO: remove try and fix that unknown Error
                            self.anchor_pts_graph_map_occupied_dots.remove()
                        except:
                            pass
                    for i, a in enumerate(self.anchor_pts_graph_map_dots_text):
                        a.remove()
                    self.anchor_pts_graph_map_dots_text[:] = []
                    for i, a in enumerate(self.anchor_pts_ogm_anchor_dots_text):
                        a.remove()
                    self.anchor_pts_ogm_anchor_dots_text[:] = []

                    if not all(self.ogm_anchor_pts_occupancy_bool):
                        self.anchor_pts_graph_map_free_dots = self.ax[1, 2].scatter(
                            self.anchor_pts_graph_map[~self.ogm_anchor_pts_occupancy_bool][:, 0] + 0.5,
                            self.anchor_pts_graph_map[~self.ogm_anchor_pts_occupancy_bool][:, 1] + 0.5,
                            facecolors='none', edgecolors='y', marker='o', s=40, linewidths=1.0, zorder=3)
                        '''# Annotattion of occupancy values
                        for i, txt in enumerate(self.ogm_anchor_pts_occupancy[~self.ogm_anchor_pts_occupancy_bool]):
                            ann = self.ax[1, 2].annotate(
                                int(txt), (self.anchor_pts_graph_map[~self.ogm_anchor_pts_occupancy_bool][i, 0] + 0.5,
                                      self.anchor_pts_graph_map[~self.ogm_anchor_pts_occupancy_bool][i, 1] + 0.5),
                                size=5, color='k', weight='bold')
                            self.anchor_pts_graph_map_dots_text.append(ann)'''
                    if any(self.ogm_anchor_pts_occupancy_bool):
                        self.anchor_pts_graph_map_occupied_dots = self.ax[1, 2].scatter(
                            self.anchor_pts_graph_map[self.ogm_anchor_pts_occupancy_bool][:, 0] + 0.5,
                            self.anchor_pts_graph_map[self.ogm_anchor_pts_occupancy_bool][:, 1] + 0.5,
                            facecolors='none', edgecolors='k', marker='o', s=40, linewidths=1.0, zorder=3)
                        # Annotattion of occupancy values
                        '''for i, txt in enumerate(self.ogm_anchor_pts_occupancy[self.ogm_anchor_pts_occupancy_bool]):
                            ann = self.ax[1, 2].annotate(
                                int(txt), (self.anchor_pts_graph_map[self.ogm_anchor_pts_occupancy_bool][i, 0] + 0.5,
                                      self.anchor_pts_graph_map[self.ogm_anchor_pts_occupancy_bool][i, 1] + 0.5),
                                size=5, color='k', weight='bold')
                            self.anchor_pts_graph_map_dots_text.append(ann)'''

                # Update OGM
                self.ogm_img_obj.set_data(self.ogm_map.map_without_dynamic_prob)
                # Annotattion of occupancy mean value
                '''for i, txt in enumerate(self.ogm_anchor_occ_sums):
                    if not txt == -1:
                        ann = self.ax_ogm.annotate(int(txt), (self.ogm_anchor_pts[i, 0], self.ogm_anchor_pts[i, 1]), size=7,
                                                   color='r', weight='bold')
                        self.anchor_pts_ogm_anchor_dots_text.append(ann)'''
                # Update title
                self.ax_ogm.set_title('LiDAR Occupancy Grid Map\nPredictions: Steer=%04.1f° %s, Throttle=%03.1f, '
                                      'Brake=%03.1f\nActual Speed=%04.1fkm/h'
                                      % (abs(control.steer) * 70., 'right' if control.steer >= 0 else 'left',
                                         control.throttle, control.brake, actual_speed * 3.6))
                # Draw vehicle position circle and shift and rotate car
                self.vpc_circle.radius = self.ogm_map.voc_radius
                if self.use_real_car_image:
                    self._shift_rotate_car_img(self.ax_ogm, self.ogm_map.pose_local)
                else:
                    for i in range(len(self.car_rects_future)):  # Blue
                        self._shift_rotate_car_rect(self.ax_ogm, self.car_rects_future[i],
                                                    self.poses_local_future_bicycle[i])
                    self._shift_rotate_car_rect(self.ax_ogm, self.car_rect, self.ogm_map.pose_local)  # Green
                # Update sensor dot
                self.sensor_dot.remove()
                self.sensor_dot = self.ax_ogm.scatter(self.ogm_map.pose_local[0], self.ogm_map.pose_local[1], c='r',
                                                      marker='o', s=20, linewidths=2)

        if (self.fig is not None) and (self.visualize or self.save_figure):
            self.fig.canvas.draw()  # update circle and car
            self.fig.canvas.flush_events()  # update figure (optional)
            """plt.figure()
            plt.hist(self.ogm_map.map_fullscan[:, 2])
            plt.show()"""

        # if frame_number == 480:
        #     plt.show()
        # plt.show(block=False)
        # plt.waitforbuttonpress()

        # plt.show()

        # Visualize route planner graph
        '''plt.figure()
        self.route_planner._city_track._map._graph.plot('r')
        plt.figure()
        self.route_planner._city_track._map._graph.plot_ori('r')'''

        if self.save_figure:
            plt.savefig(r"C:\\Work\\Software\\CARLA\\results\\ogm\\" + str(frame_number) + '.png', dpi=80,
                        bbox_inches='tight')
        else:
            plt.show(block=False)

        if self.high_quality_frame:  # high quality snap for the paper
            plt.savefig(r"C:\\Work\\Software\\CARLA\\results\\ogm\\" + str(frame_number) + '.png', dpi=100,
                        bbox_inches='tight')
            print("Actual speed = %f km/h." % (actual_speed * 3.6))
            return

    def _shift_rotate_car_img(self, ax, pose_local):
        x1, x2, y1, y2 = self.car_img.get_extent()
        self.car_img._image_skew_coordinate = (x2, y1)
        # center_x, center_y = self.car_length_on_map // 2, self.car_width_on_map // 2
        sensor_x, sensor_y = self.car_length_on_map / 2, self.car_width_on_map / 2
        img_trans = \
            mtransforms.Affine2D().rotate_deg_around(sensor_x, sensor_y, -math.degrees(pose_local[2])).translate(
                pose_local[0] - sensor_x, pose_local[1] - sensor_y) + ax.transData
        self.car_img.set_transform(img_trans)

    def _shift_rotate_car_rect(self, ax, rect, pose_local):
        sensor_x, sensor_y = self.car_length_on_map / 2, self.car_width_on_map / 2
        trans = mtransforms.Affine2D().rotate_deg_around(sensor_x, sensor_y, -math.degrees(pose_local[2])).translate(
            pose_local[0] - sensor_x, pose_local[1] - sensor_y) + ax.transData
        rect.set_transform(trans)