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
from planner.waypointer import Waypointer
import copy
import scipy

class OGM_Planner():
    def __init__(self, town_name, simulator_fps, route_planner, planner_cells_to_waypoints=10,
                 visualize_ogm_planner=False, save_ogm_planner_figure=False, start_visualize_or_save_from_frame=0,
                 normal_save_quality=50, save_high_quality_frame_numbers=[], visualize_save_directory=""):
        # Visualization settings
        self.start_visualize_or_save_from_frame = start_visualize_or_save_from_frame # start consuming time from that frame
        self.normal_save_quality = normal_save_quality  # 50, 400 for decent quality
        self.save_high_quality_frame_numbers = save_high_quality_frame_numbers  # [434, 522, 864] on ex = 4 and pos = 9 CoRL benchamrk, list(range(500, 900))
        self.visualize = visualize_ogm_planner
        self.save_figure = save_ogm_planner_figure
        self.visualize_ogm_only = False
        self.use_real_car_image = False  # In the OGM displayed ego-car
        self.visualize_save_directory = visualize_save_directory

        # Parameters for car
        self.ogm_map = OGM()
        self.car_width = 2.0 * 0.9399999976158142  # from measurements.player_measurements.bounding_box.extent for the default ego car: Mustang [in meters]
        self.car_length = 2.0 * 2.3399999141693115  # from measurements.player_measurements.bounding_box.extent for the default ego car: Mustang [in meters]

        # Parameters for bicycle model and rectify steering based on OGM
        self.wheel_base = 2.89  # for the default ego car: Mustang
        self.wheel_radius = 0.32  # for the default ego car: Mustang, not needed for our simple bicycle model
        self.poses_local_future_bicycle = -1  # unset
        self.ogm_future_trajectory = -1
        self.frames_in_future_bicycle = 50  # 50, after which steering will be 0 (45 means 3 seconds in future if self.simulator_fps=15)
        self.frames_skip_bicycle = 10  # 2, self.frames_in_future_bicycle should multiple of it
        self.trajectory_length_after_resampling = 60  # 60
        self.trajectory_max_distance = 5  # 5, remove points that are more far than this (+ points in car length)
        self.ogm_rectify_rerouting_window = 2  # 2 (TODO 2 or 3, which is better?) is chosen [OGM cells] as good value, should be less with lower self.frames_skip_bicycle (more dense future points)
        self.ogm_rectify_step = 0.01  # 0.01 rectify steering based on OGM (percentage of current steer) steering is from -1 to 1 (70 degrees right)
        self.ogm_rectify_numer_of_steps = 40  # 40, steps of trials of rectifying steering based on OGM in each direction
        self.ogm_ctrl_rectification_occupacny_th = 0.51  # 0.51

        # Parameters for making OGM reroute route planner
        self.number_of_cells_ahead_to_consider = 3  # 4, planner graph cells ahead the car cell to consider for occupancy (inserted wall is by maximum number_of_cells_ahead_to_consider cells ahead of the car current cell)
        self.ogm_route_occupacny_th = 0.70  # 0.55, a probability from 0 to 1
        self.ogm_rerouting_window = 4 # 4 is a chosen value [OGM cells] for width&height=80m, resolution=0.5m, planner_cells_to_waypoints=12, bezier nTimes (total_course_trajectory_distance/10 in waypointer.py). Should be small given dense bezier output curve points
        self.ogm_route_occupacny_th1 = 400  # 400 is chosen, cell is occupied if, its waypoints have more occupied OGM pixels

        self.visualization_scale_for_town1 = 0.5  # ratio to bigger scale of 1.0 in town 2, because town2 planner grid map is around the double of town1 one
        self.simulator_fps = simulator_fps
        self.planner_current_pos_GPS = None
        self.planner_route_cells = None
        self.planner_route_cells_OGM_ahead = None
        self.planner_route_cells_pixel_map_ahead = None
        self.planner_route_cells_occupied = None  # True for occupied route cells detected from OGM
        self.planner_route_cells_occupancy = None  # The occupancy values itself for visualization
        self.last_added_walls = []
        self.previous_a_star_cells = None
        self.previous_a_star_route = None
        self.previous_next_node = None
        self.added_walls = []
        # self.added_walls_age = []  # frame number of the added wall, to remove it after some time

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

        self.fig = None  # to open a single figure visualization
        self.vpc_circle = None  # for vehicle position circle visualization
        self.pointcloud_scatter = None  # for scan points visualization
        self.sensor_dot = None  # for laser sensor location point visualization
        self.ax_ogm = None
        self.ogm_future_trajectory_scatter = None
        self.planner_route_cells_OGM_scatter = None
        self.planner_route_cells_pixel_map_ahead_scatter = None
        self.planner_route_cells_occupied_scatter = None
        self.planner_route_cells_text = []
        self.planner_astar_undirected_walls_scatter = None
        self.planner_astar_directed_walls_scatter = None
        self.planner_astar_route_scatter = None
        self.travelled_trajectory_pixel_map = []
        self.travelled_trajectory_graph_map = []
        self.travelled_trajectory_pixel_map_scatter = None
        self.travelled_trajectory_graph_map_scatter = None

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
        self.planner_cells_to_waypoints = planner_cells_to_waypoints
        self.waypointer = Waypointer(town_name, planner_cells_to_waypoints)

        if town_name == 'Town01':
            self.visualization_scale = self.visualization_scale_for_town1
        else:
            self.visualization_scale = 1

    # return the nearest road node regardless how far
    def GPS_to_planner_nearest_road_node(self, GPS_pos):
        # use "_city_track.get_map().convert_to_node" or "_city_track.project_node" to give the neares road (non-wall) cell)
        return self.route_planner._city_track.project_node(GPS_pos, use_max_scale_of=-1)

    # returns (-1,-1) if the GPS isn't road (not used in current code version)
    """def GPS_to_planner_road_node(self, GPS_pos):
        # use "_city_track.get_map().convert_to_node" or "_city_track.project_node" to give the neares road (non-wall) cell)
        return self.route_planner._city_track.project_node(GPS_pos, use_max_scale_of=0)  # important parameter, 0 means return direct road nodes only, everything else is returned (-1,-1)"""

    def set_source_destination_from_GPS(self, source_pos, destination_pos):
        planner_current_pos_img_world_GPS = [source_pos.location.x, source_pos.location.y, 0.22]
        self.planner_destination_pos_img_pixel_map = [destination_pos.location.x, destination_pos.location.y, 0.22]

        # Convert GPS to Nodes map coordinates
        self.planner_source_pos_graph_map = self.GPS_to_planner_nearest_road_node(planner_current_pos_img_world_GPS)
        self.planner_destination_pos_graph_map = self.GPS_to_planner_nearest_road_node(self.planner_destination_pos_img_pixel_map)

        # Convert GPS to world colored map pixels coordinates
        self.planner_source_pos_img_pixel_map = self.route_planner._city_track.get_map().convert_to_pixel(planner_current_pos_img_world_GPS)
        self.planner_destination_pos_img_pixel_map = self.route_planner._city_track.get_map().convert_to_pixel(self.planner_destination_pos_img_pixel_map)

    def find_closest_planner_route_cells_centres_OGM(self, node):
        dist_2 = np.sum((self.planner_route_cells_centres_OGM - node) ** 2, axis=1)
        return np.argmin(dist_2)

    # current_pos_transform_GPS has the GPS position of the car
    # control is the predicted control from NN
    # control.steer is from -1 to 1, the maximum for steering of the used ego car (Mustang) is 70 degrees
    # planner_command_input: LANE_FOLLOW = 2.0, REACH_GOAL = 0.0, TURN_LEFT = 3.0, TURN_RIGHT = 4.0, GO_STRAIGHT = 5.0
    # returns control as is, or modify it if needed
    def step(self, sensor_data, img_pgm, measurements, control, planner_command_input, current_pos_transform_GPS,
             target_pos_transform_GPS, enable_steering_rect_using_OGM, first_step, planner_route_current_cell):
        '''if measurements.frame_number > 1000:  # To stop profiler during optimization
            exit()'''

        # Get some measurements
        self.planner_current_pos_GPS = [current_pos_transform_GPS.location.x, current_pos_transform_GPS.location.y, 0.22]

        self.planner_current_pos_graph_map = self.GPS_to_planner_nearest_road_node(self.planner_current_pos_GPS)  # Convert GPS to Nodes map coordinates
        self.planner_current_pos_img_pixel_map = self.route_planner._city_track.get_map().convert_to_pixel(self.planner_current_pos_GPS)  # Convert GPS to world colored map pixels coordinates

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

        # If first step remove all OGM added walls
        if first_step:
            self.route_planner._city_track.OGM_occupied.clear()

        # Check if there is no possible route after adding new walls from OGM, remove them and use old route instead
        if not self.route_planner._city_track.astar_route:
            for w in self.last_added_walls:
                self.route_planner._city_track.OGM_occupied.discard(w)
                self.added_walls.pop()
                # self.added_walls_age.pop()
            if self.previous_a_star_cells is not None:
                self.route_planner._city_track.astar_cells = self.previous_a_star_cells
                self.route_planner._city_track.astar_route = self.previous_a_star_route
                self.route_planner._next_node = self.previous_next_node

        # Get route planner cells and convert them to waypoints
        self.planner_route_cells = np.array([*self.route_planner._city_track.astar_route])  # For visualization
        self.planner_route_cells_ahead = np.array([*self.route_planner._city_track.astar_route[planner_route_current_cell:]])  # calculated route from car to future
        planner_route_cells_GPS_ahead, self.planner_route_cells_pixel_map_ahead, _ = self.waypointer.get_next_waypoints(
            self.planner_route_cells_ahead,
            (current_pos_transform_GPS.location.x,
             current_pos_transform_GPS.location.y, 0.22),
            (current_pos_transform_GPS.orientation.x,
             current_pos_transform_GPS.orientation.y,
             current_pos_transform_GPS.orientation.z),
            (target_pos_transform_GPS.location.x, target_pos_transform_GPS.location.y, 0.22),
            (target_pos_transform_GPS.orientation.x, target_pos_transform_GPS.orientation.y,
             target_pos_transform_GPS.orientation.z))
        planner_route_cells_GPS_ahead = np.array(planner_route_cells_GPS_ahead[::-1])  # reverse, because get_next_waypoints returns it reversed
        self.planner_route_cells_pixel_map_ahead = np.array(self.planner_route_cells_pixel_map_ahead)

        # Project waypoitns to OGM (and pixel world map), Coordinates available are (represented as x then y): self.ogm_map.pose_local, self.planner_current_pos_GPS, self.planner_current_pos_img_pixel, self.planner_current_pos_graph_map
        self.planner_route_cells_OGM_ahead = np.zeros((self.planner_route_cells_pixel_map_ahead.shape))
        self.planner_route_cells_occupancy = np.zeros((self.planner_route_cells.shape[0]))  # array of zeros of length self.planner_route_cells
        self.planner_route_cells_occupied = np.zeros((self.planner_route_cells.shape[0]), dtype=bool)  # array of Falses of length self.planner_route_cells
        self.planner_route_cells_centres_OGM = np.array(self.planner_route_cells_ahead)
        if len(planner_route_cells_GPS_ahead.shape)>1:  # Doesn't happen towards reaching goal
            self.planner_route_cells_OGM_ahead[:, 0] = (self.ogm_map.pose_local[0] * self.ogm_map.cfg.resolution +
                                                  planner_route_cells_GPS_ahead[:, 0] - self.planner_current_pos_GPS[0] ) / self.ogm_map.cfg.resolution
            self.planner_route_cells_OGM_ahead[:, 1] = (self.ogm_map.pose_local[1] * self.ogm_map.cfg.resolution +
                                                  planner_route_cells_GPS_ahead[:, 1] - self.planner_current_pos_GPS[1]) / self.ogm_map.cfg.resolution

            # Update occupancy for each planner cell, a cell is occupied when at least one of each waypoints has a high occupancy around it
            waypoints_idx_inside_ogm = np.logical_and(
                np.logical_and(self.planner_route_cells_OGM_ahead[:, 0] >= 0, self.planner_route_cells_OGM_ahead[:, 1] >= 0),
                np.logical_and(self.planner_route_cells_OGM_ahead[:, 0] < self.ogm_map.map.shape[0],
                               self.planner_route_cells_OGM_ahead[:, 1] < self.ogm_map.map.shape[1]))
            step = int(np.floor(self.planner_route_cells_OGM_ahead.shape[0] / self.planner_cells_to_waypoints))  # assuming waypoints are equally distributed on the asked route graph cells
            # Convert graph cells to OGM coordinates (self.planner_route_cells_centres_OGM)
            GPS = np.apply_along_axis(self.route_planner._city_track._map._converter._node_to_world, 1, self.planner_route_cells_ahead)
            self.planner_route_cells_centres_OGM[:, 0] = (self.ogm_map.pose_local[0] * self.ogm_map.cfg.resolution + GPS[:, 0]
                                                          - self.planner_current_pos_GPS[0]) / self.ogm_map.cfg.resolution
            self.planner_route_cells_centres_OGM[:, 1] = (self.ogm_map.pose_local[1] * self.ogm_map.cfg.resolution + GPS[:, 1]
                                                          - self.planner_current_pos_GPS[1]) / self.ogm_map.cfg.resolution
            # Associate self.planner_route_cells_OGM_ahead (waypoints) to nearest self.planner_route_cells_centres_OGM (graoh cells)
            indices = np.apply_along_axis(self.find_closest_planner_route_cells_centres_OGM, 1, self.planner_route_cells_OGM_ahead)
            for i, cell in enumerate(self.planner_route_cells_ahead[1:]):  # For each planner graph path cell
                if i == 0: # current car position cell, keep occupancy of 0 always
                    continue
                OGM_points_within = self.planner_route_cells_OGM_ahead[indices == i]
                OGM_points_within = OGM_points_within[waypoints_idx_inside_ogm[indices == i]]
                occupancies = []
                for c in OGM_points_within:
                    row_from = int(max(0, c[1] - self.ogm_rerouting_window))
                    row_to = int(min(self.ogm_map.map_without_dynamic_prob.shape[0], c[1] + self.ogm_rerouting_window))
                    col_from = int(max(0, c[0] - self.ogm_rerouting_window))
                    col_to = int(min(self.ogm_map.map_without_dynamic_prob.shape[1], c[0] + self.ogm_rerouting_window))
                    OGM_slice = self.ogm_map.map_without_dynamic_prob[row_from:row_to, col_from:col_to]
                    occupancies.append(len(OGM_slice[OGM_slice > self.ogm_route_occupacny_th]))  # Important factor; mean, max, ...
                if occupancies:
                    self.planner_route_cells_occupancy[i+planner_route_current_cell] = np.sum(occupancies)  # Important factor; mean, max, ...
                if self.planner_route_cells_occupancy[i+planner_route_current_cell] > self.ogm_route_occupacny_th1:
                    self.planner_route_cells_occupied[i+planner_route_current_cell] = True
            # Delete out of OGM points (for visualization)
            self.planner_route_cells_OGM_ahead = self.planner_route_cells_OGM_ahead[waypoints_idx_inside_ogm]

        # Add walls to occupied planner grid cells
        # TODO: added walls should be directed to allow the other lane to be free and to prevent looping in case of removal after vehcile leaves the area
        self.previous_a_star_cells = copy.deepcopy(self.route_planner._city_track.astar_cells)
        self.previous_a_star_route = copy.deepcopy(self.route_planner._city_track.astar_route)
        self.previous_next_node = copy.deepcopy(self.route_planner._next_node)
        self.last_added_walls = []
        for i, occ in enumerate(self.planner_route_cells_occupied):
            if (i-planner_route_current_cell) <= self.number_of_cells_ahead_to_consider and occ:
                len_before = len(self.route_planner._city_track.OGM_occupied)
                self.route_planner._city_track.OGM_occupied.add(tuple(self.planner_route_cells[i, :]))
                if len_before != len(self.route_planner._city_track.OGM_occupied):  # There are new walls this step
                    self.route_planner.OGM_route_reroute = True
                    self.last_added_walls.append(tuple(self.planner_route_cells[i, :]))
                    self.added_walls.append(tuple(self.planner_route_cells[i, :]))
                    # self.added_walls_age.append(frame_number)
            # TODO: invistigate if needed in some scenarios
            ''' else:
                set.remove'''

        # Predict future pose using bicycle model
        # self.ogm_map.pose_local[2] = math.radians(70)  # self.ogm_map.pose_local: [cells], [cells], [radians counter clockwise]
        # steering_angle = 0.3  # -1:1 (-70 to 70 degrees), 1 means 70 degrees to the car right
        # actual_speed = 20  # m/s
        # self.frames_in_future_bicycle = 15
        self.poses_local_future_bicycle, self.ogm_future_trajectory = \
            self._bicycle_model(self.ogm_map.pose_local, control.steer, actual_speed, self.frames_in_future_bicycle,
                                self.frames_skip_bicycle)

        # Recify model predicted controls using motion model (kinematic bicycle model)
        OGM_rectified_control = copy.deepcopy(control)
        if enable_steering_rect_using_OGM:
            steps = 0
            good_steer = False
            while steps < self.ogm_rectify_numer_of_steps and not good_steer:  # rectify steering on the same steering direction (sign)
                debugging = False
                if debugging:
                    print(frame_number)
                    if frame_number == 312:  # For debugging
                        i = 0
                        row_from = int(max(0, self.ogm_future_trajectory[i, 1] - self.ogm_rectify_rerouting_window))
                        row_to = int(min(self.ogm_map.map_without_dynamic_prob.shape[0],
                                         self.ogm_future_trajectory[i, 1] + self.ogm_rectify_rerouting_window))
                        col_from = int(max(0, self.ogm_future_trajectory[i, 0] - self.ogm_rectify_rerouting_window))
                        col_to = int(min(self.ogm_map.map_without_dynamic_prob.shape[1],
                                         self.ogm_future_trajectory[i, 0] + self.ogm_rectify_rerouting_window))
                        OGM_slice = self.ogm_map.map_without_dynamic_prob[row_from:row_to,
                                    col_from:col_to]  # TODO: self.ogm_map.map_without_dynamic_prob or map ?
                        plt.figure()
                        plt.imshow(OGM_slice)
                        self.visualize = True
                        self.save_figure = False
                        self._visualize_planning(frame_number, actual_speed, control, OGM_rectified_control,
                                                 planner_command_input, fullscan, points_semseg_color, img, img_pgm,
                                                 img_semseg, visualization_step=1)  # increase visualization_step for faster debugging

                steps += 1
                good_steer = True  # assume and the following for loop will verify
                for i in range(self.ogm_future_trajectory.shape[0]):
                    row_from = int(max(0, self.ogm_future_trajectory[i, 1] - self.ogm_rectify_rerouting_window))
                    row_to = int(min(self.ogm_map.map_without_dynamic_prob.shape[0], self.ogm_future_trajectory[i, 1] + self.ogm_rectify_rerouting_window))
                    col_from = int(max(0, self.ogm_future_trajectory[i, 0] - self.ogm_rectify_rerouting_window))
                    col_to = int(min(self.ogm_map.map_without_dynamic_prob.shape[1], self.ogm_future_trajectory[i, 0] + self.ogm_rectify_rerouting_window))
                    OGM_slice = self.ogm_map.map_without_dynamic_prob[row_from:row_to, col_from:col_to]  # TODO: self.ogm_map.map_without_dynamic_prob or map ?
                    if len(OGM_slice[OGM_slice > self.ogm_ctrl_rectification_occupacny_th]) > 0:  # TODO: > 0 or another threshold?
                        # steer is from 1 (70 degrees right) to -1
                        new_steer = OGM_rectified_control.steer + np.sign(OGM_rectified_control.steer) * self.ogm_rectify_step  # plus
                        if abs(new_steer) <= 1:
                            OGM_rectified_control.steer = new_steer
                            self.poses_local_future_bicycle, self.ogm_future_trajectory = \
                                self._bicycle_model(self.ogm_map.pose_local, OGM_rectified_control.steer, actual_speed,
                                                    self.frames_in_future_bicycle, self.frames_skip_bicycle)
                        good_steer = False
                        break
            if not good_steer:  # Failed
                # print("----- Frame %d: Tried making rectifying steer in same direction but failed" % (frame_number))
                OGM_rectified_control = copy.deepcopy(control)
                self.poses_local_future_bicycle, self.ogm_future_trajectory = \
                    self._bicycle_model(self.ogm_map.pose_local, control.steer, actual_speed,
                                        self.frames_in_future_bicycle,
                                        self.frames_skip_bicycle)
            steps = 0
            while steps < self.ogm_rectify_numer_of_steps and not good_steer:  # If failed, rectify steering on the oppissite direction
                steps += 1
                good_steer = True  # assume and the following for loop will verify
                for i in range(self.ogm_future_trajectory.shape[0]):
                    row_from = int(max(0, self.ogm_future_trajectory[i, 1] - self.ogm_rectify_rerouting_window))
                    row_to = int(min(self.ogm_map.map_without_dynamic_prob.shape[0], self.ogm_future_trajectory[i, 1] + self.ogm_rectify_rerouting_window))
                    col_from = int(max(0, self.ogm_future_trajectory[i, 0] - self.ogm_rectify_rerouting_window))
                    col_to = int(min(self.ogm_map.map_without_dynamic_prob.shape[1], self.ogm_future_trajectory[i, 0] + self.ogm_rectify_rerouting_window))
                    OGM_slice = self.ogm_map.map_without_dynamic_prob[row_from:row_to, col_from:col_to]  # TODO: self.ogm_map.map_without_dynamic_prob or map ?
                    if len(OGM_slice[OGM_slice > self.ogm_ctrl_rectification_occupacny_th]) > 0:  # TODO: > 0 or another threshold?
                        # steer is from 1 (70 degrees right) to -1
                        new_steer = OGM_rectified_control.steer - np.sign(OGM_rectified_control.steer) * self.ogm_rectify_step  # minus
                        if abs(new_steer) <= 1:
                            OGM_rectified_control.steer = new_steer
                            self.poses_local_future_bicycle, self.ogm_future_trajectory = \
                                self._bicycle_model(self.ogm_map.pose_local, OGM_rectified_control.steer, actual_speed,
                                                    self.frames_in_future_bicycle, self.frames_skip_bicycle)
                        good_steer = False
                        break
            if not good_steer:  # Both failed (most probably when the car or a part of it is on the sidewalk already)
                # print("----- Frame %d: Tried making rectifying steer in opposite direction but failed" % (frame_number))
                OGM_rectified_control = copy.deepcopy(control)
                self.poses_local_future_bicycle, self.ogm_future_trajectory = \
                    self._bicycle_model(self.ogm_map.pose_local, control.steer, actual_speed,
                                        self.frames_in_future_bicycle,
                                        self.frames_skip_bicycle)
            '''if control.steer != OGM_rectified_control.steer:
                print("----- Frame %d: Steer rectified by %08.5f° %s" %
                      (frame_number, 
                      0 if (control.steer - OGM_rectified_control.steer) == 0 else 
                      abs(control.steer + OGM_rectified_control.steer) * 70.
                      if sign(control.steer) != sign(OGM_rectified_control.steer) else
                      abs(control.steer - OGM_rectified_control.steer) * 70.,
                       '' if (control.steer - OGM_rectified_control.steer) == 0 else 'left'
                       if (control.steer - OGM_rectified_control.steer) >= 0 else 'right'))'''

        # Visualize Map
        self.travelled_trajectory_pixel_map.append(self.planner_current_pos_img_pixel_map)
        self.travelled_trajectory_graph_map.append(self.planner_current_pos_graph_map)
        self._visualize_planning(frame_number, actual_speed, control, OGM_rectified_control, planner_command_input,
                                 fullscan, points_semseg_color, img, img_pgm, img_semseg,
                                 visualization_step=1)  # increase visualization_step for faster debugging

        return OGM_rectified_control

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

    # Assumes that speed will decay to zero in frames_in_future
    # pose_local = [x,y of vehicle centre point in map cells (not meters), yaw in radians counter-clockwise]
    # steering_angle: tire steering angle is from -1:1 (-70 to 70 degrees), 1 means 70 degrees to the car right (extreme angle #TODO: as in model training data)
    # actual_speed is in m/s
    # steering_angle is from -1 (extrmete left) to 1 (extrmete right), the maximum for steering of the used ego car (Mustang) is 70 degrees
    def _bicycle_model(self, pose_local, steering_angle, actual_speed, frames_in_future, frames_to_skip):
        steering_angle = -steering_angle * math.radians(70)
        seconds_dt = frames_to_skip / self.simulator_fps

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

        # Equidistant resampling and smooth trajectory
        ogm_future_trajectory = np.vstack((self.ogm_map.pose_local, poses_local_future_bicycle))
        x = ogm_future_trajectory[:, 0]
        y = ogm_future_trajectory[:, 1]
        distances = np.cumsum(np.sqrt(np.ediff1d(x, to_begin=0) ** 2 + np.ediff1d(y, to_begin=0) ** 2))
        if np.isnan(distances).any():  # Happens when poses_local_future_bicycle is the same point (car actual speed = 0)
            ogm_future_trajectory = np.tile(self.ogm_map.pose_local[0:2], (self.trajectory_length_after_resampling, 1))
        else:
            distances = distances / distances[-1]
            fx, fy = scipy.interpolate.interp1d(distances, x), scipy.interpolate.interp1d(distances, y)
            alpha = np.linspace(0, 1, self.trajectory_length_after_resampling)
            x_new, y_new = fx(alpha), fy(alpha)
            ogm_future_trajectory = np.column_stack((x_new, y_new))
            if np.isnan(ogm_future_trajectory).any():  # Happens when poses_local_future_bicycle is the same point (car actual speed = 0) (might be not needed here because checked before already)
                ogm_future_trajectory = np.tile(self.ogm_map.pose_local[0:2], (self.trajectory_length_after_resampling, 1))

        # Remove far points and points within car body
        x = ogm_future_trajectory[:, 0]
        y = ogm_future_trajectory[:, 1]
        distances = np.cumsum(np.sqrt(np.ediff1d(x, to_begin=0) ** 2 + np.ediff1d(y, to_begin=0) ** 2))
        distances = distances * self.ogm_map.cfg.resolution - self.car_length / 2 # gives points distances from car front in meters
        ogm_future_trajectory = ogm_future_trajectory[(distances > 0) & (distances < self.trajectory_max_distance), :]

        return poses_local_future_bicycle, ogm_future_trajectory

    # TODO: clear planning map access after episode/pose/experiment ends
    def _visualize_planning(self, frame_number, actual_speed, control, OGM_rectified_control, planner_command_input,
                            fullscan, points_semseg_color, img, img_pgm, img_semseg, visualization_step=1):
        print('Visualized Frame Number: %d'%(frame_number))
        if (not self.fig) or (frame_number % visualization_step) == 0 and frame_number >= self.start_visualize_or_save_from_frame:  # Replace the last condition with another one to visualize for debugging
            if self.visualize or self.save_figure:
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
                        self.ax[1, 0].set_title('LiDAR PGM')
                        self.ax[0, 2].set_title('World Map')
                        self.ax[1, 2].set_title('Route Planner Map (one way roads)\nCurrent Command: %s' % (planner_command_input))
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
                                                   markersize=12, markeredgecolor=[0., 0., 0.]) for c in  [[0., 1., 0.]]]
                        markers.extend([Line2D([0], [0], marker='s', color='w', label='Scatter', markerfacecolor=c,
                                          markersize=12, markeredgecolor=[0., 0., 0.]) for c in [[1., 1., 1.]]])
                        markers.extend([Line2D([0], [0], marker='s', color='w', label='Scatter', markerfacecolor=c,
                                          markersize=12, markeredgecolor='w') for c in [[0., 0., 0.]]])
                        markers.extend([Line2D([], [], marker='o', color='w', markeredgecolor=c, label='Scatter',
                                               markerfacecolor='w', markersize=10) for c in ['c']])
                        markers.extend([Line2D([], [], marker='.', color='w', markeredgecolor=c, label='Scatter',
                                               markerfacecolor=c, markersize=6) for c in [[1., 1., 0.]]])
                        markers.extend([Line2D([0], [0], marker='s', color='w', label='Scatter', markerfacecolor=c,
                                               markersize=12, markeredgecolor=[0., 0., 0.]) for c in [[1., 0., 0.]]])
                        markers.extend([Line2D([], [], marker='.', color='w', markeredgecolor=c, label='Scatter',
                                               markerfacecolor=c, markersize=6) for c in [[0., 0., 1.]]])
                        self.ax_ogm.legend(markers, ['Ego-vehicle', 'Occupied', 'Free', 'Positioning Circle',
                                                     'Planner\nWaypoints', 'Motion Model', 'Rectified\nTrajectory'],
                                           loc='center left', bbox_to_anchor=(1.01, 0.5))
                        self.ax[1, 1].legend([Line2D([0], [0], marker='o', color='w', label='Scatter', markerfacecolor=c,
                                                     markersize=10) for c in self.classes_colors / 255.],
                                             self.classes_labels, loc='center left', bbox_to_anchor=(1.01, 0.5))
                        markers = [Line2D([0], [0], marker='o', color='w', label='Scatter', markerfacecolor=c,
                                                     markersize=10) for c in [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]]
                        markers.extend([Line2D([], [], marker='.', color='w', markeredgecolor=c, label='Scatter',
                                               markerfacecolor=c, markersize=6) for c in [[0, 1., 1.]]])
                        markers.extend([Line2D([], [], marker='.', color='w', markeredgecolor=c, label='Scatter',
                                               markerfacecolor=c, markersize=6) for c in [[1., 1., 0.]]])
                        self.ax[0, 2].legend(markers,
                                             ['Source', 'Destination', 'Current', 'Travelled\nTrajectory',
                                              'Planner\nWaypoints'], loc='center left', bbox_to_anchor=(1.01, 0.5))
                        markers = [Line2D([0], [0], marker='o', color='w', label='Scatter', markerfacecolor=c,
                                          markersize=10) for c in [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]]
                        markers.extend([Line2D([], [], marker='.', color='w', markeredgecolor=c, label='Scatter',
                                               markerfacecolor=c, markersize=5) for c in [[0., 1., 1.]]])
                        markers.extend([Line2D([], [], marker='s', color='w', markeredgecolor='w', label='Scatter',
                                               markerfacecolor='#D3D3D3', markersize=10)])
                        markers.extend([Line2D([], [], marker='x', color='w', markeredgecolor=c, label='Scatter',
                                               markerfacecolor='w', markersize=7) for c in [[0., 0., 0.]]])
                        markers.extend([Line2D([], [], marker='x', color='w', markeredgecolor=c, label='Scatter',
                                               markerfacecolor='w', markersize=7) for c in [[1., 0., 0.]]])
                        markers.extend([Line2D([], [], marker='.', color='w', markeredgecolor=c, label='Scatter',
                                               markerfacecolor=c, markersize=8) for c in [[1., 0.5, 0.]]])
                        markers.extend([Line2D([], [], marker='o', color='w', markeredgecolor=c, label='Scatter',
                                               markerfacecolor='w', markersize=8) for c in [[0., 0., 0.]]])
                        self.ax[1, 2].legend(markers, ['Source', 'Destination', 'Current', 'Travelled Cells',
                                                       'Off-road Cells', 'Walls', 'Directed Walls','A* Route Cells', 'OGM Occupied'],
                                             loc='center left', bbox_to_anchor=(1.01, 0.5))

                        # Display camera and semseg images
                        self.img_obj = self.ax[0, 0].imshow(img)
                        self.ax[0, 0].axis('off')
                        self.img_pgm_obj = self.ax[1, 0].imshow(img_pgm.astype(int))  # img_semseg or img_pgm, based on which you want to visualize
                        self.ax[1, 0].axis('off')
                        # Visualize pointcloud
                        self.pointcloud_scatter = self.ax[1, 1].scatter(
                            fullscan[:, 0], -fullscan[:, 1], c=np.array(points_semseg_color) / 255., marker='.',
                            s=3.0, linewidths=0.4)  # alpha=0.7

                        # Display route planner world colored map
                        self.planner_world_map = self.ax[0, 2].imshow(
                            self.route_planner._city_track.get_map().map_image, origin='upper')
                        self.planner_source_pos_img_pixel_map_dot = self.ax[0, 2].scatter(
                            self.planner_source_pos_img_pixel_map[0], self.planner_source_pos_img_pixel_map[1],
                            c='r', marker='o', s=20, linewidths=2, zorder=2)
                        self.planner_destination_pos_img_pixel_map_dot = self.ax[0, 2].scatter(
                            self.planner_destination_pos_img_pixel_map[0], self.planner_destination_pos_img_pixel_map[1],
                            c='g', marker='o', s=20, linewidths=2, zorder=2)
                        self.planner_current_pos_img_pixel_map_dot = self.ax[0, 2].scatter(
                            self.planner_current_pos_img_pixel_map[0], self.planner_current_pos_img_pixel_map[1],
                            c='b', marker='o', s=20, linewidths=3)
                        self.planner_route_cells_pixel_map_ahead_scatter = self.ax[0, 2].scatter(
                            self.planner_route_cells_pixel_map_ahead[:, 0], self.planner_route_cells_pixel_map_ahead[:, 1],
                            c='y', marker='.', s=10, linewidths=1)
                        self.ax[0, 2].set_yticks([])
                        self.ax[0, 2].set_xticks([])

                        # Display route planner graph map, show grid lines to indicate quantization of the nodes grid map (Note: x and y axis are exchanged)
                        from matplotlib import colors as mplcolors
                        cMap = mplcolors.ListedColormap(['w', '#D3D3D3'])  # '#D3D3D3' is light gray
                        self.ax[1, 2].pcolormesh(self.route_planner._city_track.get_map()._grid._structure.T,
                                                 edgecolors='k', linewidth=0.05 ,cmap=cMap, antialiased=True)
                        self.ax[1, 2].invert_yaxis()
                        self.ax[1, 2].set_aspect('equal')
                        self.ax[1, 2].set_yticks([])
                        self.ax[1, 2].set_xticks([])
                        self.planner_source_pos_graph_map_dot = self.ax[1, 2].scatter(
                            self.planner_source_pos_graph_map[0] + 0.5, self.planner_source_pos_graph_map[1] + 0.5,
                            c='r', marker='o', s=20*self.visualization_scale, linewidths=2, zorder=4)
                        self.planner_destination_pos_graph_map_dot = self.ax[1, 2].scatter(
                            self.planner_destination_pos_graph_map[0] + 0.5, self.planner_destination_pos_graph_map[1] + 0.5,
                            c='g', marker='o', s=20*self.visualization_scale, linewidths=2, zorder=4)
                        self.planner_current_pos_graph_map_dot = self.ax[1, 2].scatter(
                            self.planner_current_pos_graph_map[0] + 0.5, self.planner_current_pos_graph_map[1] + 0.5,
                            c='b', marker='o', s=20*self.visualization_scale, linewidths=2, zorder=4)
                        walls = [x for x in self.route_planner._city_track.astar_cells if not x.reachable]
                        if walls:
                            self.planner_astar_undirected_walls_scatter = self.ax[1, 2].scatter(
                                [i.x + 0.5 for i in walls], [i.y + 0.5 for i in walls],
                                c='k', marker='x', s=20 * self.visualization_scale, linewidths=1, zorder=6)
                        if self.planner_route_cells is not None:
                            self.planner_astar_route_scatter = self.ax[1, 2].scatter(
                                self.planner_route_cells[:, 0] + 0.5, self.planner_route_cells[:, 1] + 0.5,
                                c=[1., 0.5, 0.], marker='.', s=50*self.visualization_scale, linewidths=1)
                        # Planner OGM occupancy
                        if any(self.planner_route_cells_occupied):
                            self.planner_route_cells_occupied_scatter = self.ax[1, 2].scatter(
                                self.planner_route_cells[self.planner_route_cells_occupied][:, 0] + 0.5,
                                self.planner_route_cells[self.planner_route_cells_occupied][:, 1] + 0.5,
                                facecolors='none', edgecolors='k', marker='o', s=40*self.visualization_scale,
                                linewidths=1.0, zorder=3)
                        # Annotattion of occupancy values
                        """for i, txt in enumerate(self.planner_route_cells_occupancy):
                            ann = self.ax[1, 2].annotate(  # '{0:.2f}'.format(txt) int(txt*100)
                                int(txt/100), (self.planner_route_cells[i, 0] + 0.5, self.planner_route_cells[i, 1] + 0.5),
                                    size=5, color='r', weight='bold', zorder=4)
                            self.planner_route_cells_text.append(ann)"""

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
                    self.ax_ogm.set_title('LiDAR OGM\nNN Predictions: Steer=%04.1f° %s, Throttle=%03.1f, '
                                          'Brake=%03.1f\nOGM Rectified Steer by: %08.5f° %s'
                                          '\nActual Speed=%04.1fkm/h'
                                          % (abs(control.steer) * 70.,
                                             '' if control.steer == 0 else 'right' if control.steer >= 0 else 'left',
                                             control.throttle, control.brake,
                                             0 if (control.steer - OGM_rectified_control.steer) == 0 else
                                             abs(control.steer + OGM_rectified_control.steer) * 70.
                                             if sign(control.steer) != sign(OGM_rectified_control.steer) else
                                             abs(control.steer - OGM_rectified_control.steer) * 70.,
                                             '' if (control.steer - OGM_rectified_control.steer) == 0 else 'left'
                                             if (control.steer - OGM_rectified_control.steer) >= 0 else 'right',
                                             actual_speed * 3.6))
                    # Draw OGM route planner cells centers
                    self.planner_route_cells_OGM_scatter = self.ax_ogm.scatter(
                        self.planner_route_cells_OGM_ahead[:,0], self.planner_route_cells_OGM_ahead[:,1],
                        c='y', marker='.', s=10, linewidths=0.5)
                    self.ogm_future_trajectory_scatter = self.ax_ogm.scatter(
                        self.ogm_future_trajectory[:, 0], self.ogm_future_trajectory[:, 1], c='b',
                        marker='.', s=4, linewidths=0.3, zorder=3)
                    # Vehicle position circle
                    self.vpc_circle = plt.Circle((self.ogm_map.map_centre_cell[0], self.ogm_map.map_centre_cell[1]), 0,
                                                 color='c', fill=False)
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
                    else:  # Draw a rectangle car
                        # Future cars bicycle model: Blue
                        self.car_rects_future = []
                        car_rects_in_future_count = len(range(0, self.frames_in_future_bicycle, self.frames_skip_bicycle))
                        for i in range(car_rects_in_future_count):
                            self.car_rects_future.append(
                                patches.Rectangle((0, 0), self.car_length_on_map, self.car_width_on_map,
                                                  linewidth=1, edgecolor='k', facecolor='r', alpha=0.50))
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
                    '''self.sensor_dot = self.ax_ogm.scatter(self.ogm_map.pose_local[0], self.ogm_map.pose_local[1],
                                                          c='r', marker='.', s=20, linewidths=2)'''

                    # Figure adjustments
                    plt.tight_layout()
                    plt.subplots_adjust(top=0.9, bottom=0.05, hspace=0.5, wspace=0.2)  # For multiplelines (long) OGM figure title, also horizontal and vertical spacing
                    # plt.show(block=False)  # To see the figure (optional)
                else:  # Not first time figure created
                    if not self.visualize_ogm_only:
                        # Update images and OGM
                        self.img_obj.set_data(img)
                        self.img_pgm_obj.set_data(img_pgm.astype(int))
                        # Remove old scan points
                        if self.pointcloud_scatter:
                            self.pointcloud_scatter.remove()
                        # Visualize pointcloud
                        """step = 10  # For faster visualization
                        idx = np.arange(0, len(fullscan), step)
                        self.scatter = self.ax[1, 1].scatter(fullscan[idx, 0], -fullscan[idx, 1],
                                                             c=np.array(points_semseg_color[idx]) / 255., marker='.', s=1.0,
                                                             linewidths=0.8)  # alpha=0.7"""
                        self.pointcloud_scatter = self.ax[1, 1].scatter(
                            fullscan[:, 0], -fullscan[:, 1], c=np.array(points_semseg_color) / 255., marker='.', s=3.0,
                            linewidths=0.4)  # alpha=0.7

                        # Update route planner world colored map
                        if self.planner_route_cells_pixel_map_ahead_scatter is not None:
                            try:  # TODO: invistigate the error while removing empty matplotlib.collections."PathCollection
                                self.planner_route_cells_pixel_map_ahead_scatter.remove()
                            except:
                                pass
                        if self.planner_route_cells_pixel_map_ahead.ndim == 2:
                            self.planner_route_cells_pixel_map_ahead_scatter = self.ax[0, 2].scatter(
                                self.planner_route_cells_pixel_map_ahead[:, 0], self.planner_route_cells_pixel_map_ahead[:, 1],
                                c='y', marker='.', s=10, linewidths=1)

                        # Update route planner graph map, Note: x and y axis are exchnaged
                        '''self.ax[1, 2].set_title('Route Planner Map (one way roads)\nCurrent Command: %s\nFar from intersection (LANE_FOLLOW): %s\nNext Intersections CMDs: %s' %
                                                (planner_command_input, "Yes" if self.route_planner.far_from_intersection else "No",
                                                 str([self.commands_chars_dict[x] for x in self.route_planner._commands])))'''
                        self.ax[1, 2].set_title('Route Planner Map (one way roads)\nCurrent Command: %s' % (planner_command_input))
                        self.planner_current_pos_img_pixel_map_dot.remove()
                        self.planner_current_pos_img_pixel_map_dot = self.ax[0, 2].scatter(self.planner_current_pos_img_pixel_map[0],
                                                                                           self.planner_current_pos_img_pixel_map[1],
                                                                                           c='b', marker='o', s=20,
                                                                                           linewidths=2, zorder=3)
                        # Draw travelled trajectory
                        if self.travelled_trajectory_pixel_map_scatter is not None:
                            self.travelled_trajectory_pixel_map_scatter.remove()
                        self.travelled_trajectory_pixel_map_scatter = self.ax[0, 2].scatter(
                            [x[0] for x in self.travelled_trajectory_pixel_map],
                            [x[1] for x in self.travelled_trajectory_pixel_map],
                            c=[0., 1., 1.], marker='.', s=10, linewidths=2, zorder=1)

                        self.planner_current_pos_graph_map_dot.remove()
                        self.planner_current_pos_graph_map_dot = self.ax[1, 2].scatter(
                            self.planner_current_pos_graph_map[0] + 0.5, self.planner_current_pos_graph_map[1] + 0.5,
                            c='b', marker='o', s=20*self.visualization_scale, linewidths=2, zorder=4)
                        # self.planner_world_map.set_data(self.route_planner._city_track.get_map().map_image)
                        # self.planner_grid_map.set_data(self.route_planner._city_track.get_map()._grid._structure)
                        self.planner_astar_route_scatter.remove()
                        if self.planner_astar_undirected_walls_scatter:
                            self.planner_astar_undirected_walls_scatter.remove()
                        if self.planner_astar_directed_walls_scatter:
                            self.planner_astar_directed_walls_scatter.remove()
                        walls = [x for x in self.route_planner._city_track.astar_cells if not x.reachable]
                        if walls:
                            self.planner_astar_undirected_walls_scatter = self.ax[1, 2].scatter(
                                [i.x + 0.5 for i in walls], [i.y + 0.5 for i in walls],
                                c='k', marker='x', s=20 * self.visualization_scale, linewidths=1, zorder=6)
                        directed_walls = \
                            [i for i in walls if ((i.x, i.y) in self.route_planner._city_track.OGM_occupied)]
                        if directed_walls:
                            self.planner_astar_directed_walls_scatter = self.ax[1, 2].scatter(
                                [i.x + 0.5 for i in directed_walls], [i.y + 0.5 for i in directed_walls],
                                c='r', marker='x', s=20 * self.visualization_scale, linewidths=1, zorder=6)
                        if self.planner_route_cells is not None:
                            self.planner_astar_route_scatter = self.ax[1, 2].scatter(
                                self.planner_route_cells[:, 0] + 0.5, self.planner_route_cells[:, 1] + 0.5,
                                c=[1., 0.5, 0.], marker='.', s=50*self.visualization_scale, linewidths=1)
                        # Travelled route cells
                        if self.travelled_trajectory_graph_map_scatter is not None:
                            self.travelled_trajectory_graph_map_scatter.remove()
                        self.travelled_trajectory_graph_map_scatter = self.ax[1, 2].scatter(
                            [x[0]+0.5 for x in self.travelled_trajectory_graph_map],
                            [x[1]+0.5 for x in self.travelled_trajectory_graph_map],
                            c=[0., 1., 1.], marker='.', s=10/self.visualization_scale, linewidths=1, zorder=3)

                        # Planner OGM occupancy
                        if self.planner_route_cells_occupied_scatter is not None:
                            try:  # TODO: invistigate the error while removing empty matplotlib.collections."PathCollection
                                self.planner_route_cells_occupied_scatter.remove()
                            except:
                                pass
                        for i, a in enumerate(self.planner_route_cells_text):
                            a.remove()
                        self.planner_route_cells_text[:] = []
                        if any(self.planner_route_cells_occupied):
                            self.planner_route_cells_occupied_scatter = self.ax[1, 2].scatter(
                                self.planner_route_cells[self.planner_route_cells_occupied][:, 0] + 0.5,
                                self.planner_route_cells[self.planner_route_cells_occupied][:, 1] + 0.5,
                                facecolors='none', edgecolors='k', marker='o', s=40*self.visualization_scale,
                                linewidths=1.0, zorder=3)
                        # Annotattion of occupancy values
                        """for i, txt in enumerate(self.planner_route_cells_occupancy):
                            ann = self.ax[1, 2].annotate(  # '{0:.2f}'.format(txt) int(txt*100)
                                int(txt/100), (self.planner_route_cells[i, 0] + 0.5, self.planner_route_cells[i, 1] + 0.5),
                                    size=5, color='r', weight='bold', zorder=4)
                            self.planner_route_cells_text.append(ann)"""

                    # Update OGM
                    self.ogm_img_obj.set_data(self.ogm_map.map_without_dynamic_prob)
                    if self.planner_route_cells_OGM_scatter is not None:
                        try:  # TODO: invistigate the error while removing empty matplotlib.collections."PathCollection
                            self.planner_route_cells_OGM_scatter.remove()
                        except:
                            pass
                    if self.planner_route_cells_OGM_ahead.ndim == 2:
                        self.planner_route_cells_OGM_scatter = self.ax_ogm.scatter(
                            self.planner_route_cells_OGM_ahead[:, 0], self.planner_route_cells_OGM_ahead[:, 1], c='y', marker='.', s=10,
                            linewidths=0.5)

                    # Update title
                    self.ax_ogm.set_title('LiDAR OGM\nNN Predictions: Steer=%04.1f° %s, Throttle=%03.1f, '
                                          'Brake=%03.1f\nOGM Rectified Steer by: %08.5f° %s'
                                          '\nActual Speed=%04.1fkm/h'
                                          % (abs(control.steer) * 70., '' if control.steer==0 else 'right' if control.steer >= 0 else 'left',
                                             control.throttle, control.brake,
                                             abs(control.steer-OGM_rectified_control.steer) * 70.,
                                             '' if (control.steer-OGM_rectified_control.steer)==0 else 'left'
                                             if (control.steer-OGM_rectified_control.steer) >= 0 else 'right',
                                             actual_speed * 3.6))
                    # Draw vehicle position circle and shift and rotate car
                    self.vpc_circle.radius = self.ogm_map.voc_radius
                    if self.use_real_car_image:
                        self._shift_rotate_car_img(self.ax_ogm, self.ogm_map.pose_local)
                    else:
                        for i in range(len(self.car_rects_future)):  # Blue
                            self._shift_rotate_car_rect(self.ax_ogm, self.car_rects_future[i],
                                                        self.poses_local_future_bicycle[i])
                        self._shift_rotate_car_rect(self.ax_ogm, self.car_rect, self.ogm_map.pose_local)  # Green
                    self.ogm_future_trajectory_scatter.remove()
                    self.ogm_future_trajectory_scatter = self.ax_ogm.scatter(self.ogm_future_trajectory[:, 0],
                                                                             self.ogm_future_trajectory[:, 1], c='b',
                                                                             marker='.', s=4, linewidths=0.3, zorder=3)
                    # Update sensor dot
                    # self.sensor_dot.remove()
                    # self.sensor_dot = self.ax_ogm.scatter(self.ogm_map.pose_local[0], self.ogm_map.pose_local[1], c='r',
                    #                                       marker='o', s=20, linewidths=2)
                self.ax_ogm.set_xlim(0, self.ogm_map.map_without_dynamic_prob.shape[1]-1)
                self.ax_ogm.set_ylim(0, self.ogm_map.map_without_dynamic_prob.shape[0]-1)
                self.ax_ogm.invert_yaxis()
                self.ax[1, 2].set_xlim(0, self.route_planner._city_track.get_map()._grid._structure.T.shape[1])
                self.ax[1, 2].set_ylim(0, self.route_planner._city_track.get_map()._grid._structure.T.shape[0])
                self.ax[1, 2].invert_yaxis()

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
                self.route_planner._city_track.get_map()._graph.plot('r')
                plt.figure()
                self.route_planner._city_track.get_map()._graph.plot_ori('r')'''

                if frame_number in self.save_high_quality_frame_numbers:  # high quality snap for a paper/documentation
                    plt.savefig(self.visualize_save_directory + str(frame_number) + '.png',
                                dpi=350,  # should be dpi=400
                                bbox_inches='tight')
                    print("Frame Number %d: Actual speed = %f km/h." % (frame_number, actual_speed * 3.6))
                elif self.save_figure:
                    plt.savefig(self.visualize_save_directory + str(frame_number) + '.png', dpi= self.normal_save_quality,  # should be dpi=50
                                bbox_inches='tight')
                else:
                    plt.show(block=False)

    def _shift_rotate_car_img(self, ax, pose_local):
        x1, x2, y1, y2 = self.car_img.get_extent()
        self.car_img._image_skew_coordinate = (x2, y1)
        # center_x, center_y = self.car_length_on_map // 2, self.car_width_on_map // 2
        sensor_x, sensor_y = self.car_length_on_map / 2, self.car_width_on_map / 2
        img_trans = \
            mtransforms.Affine2D().rotate_deg_around(sensor_x, sensor_y, -math.degrees(pose_local[2])).translate(
                pose_local[0] - sensor_x, pose_local[1] - sensor_y) + ax.transData
        self.car_img.set_transform(img_trans)

    # pose_local is car centre point
    def _shift_rotate_car_rect(self, ax, rect, pose_local):
        sensor_x, sensor_y = self.car_length_on_map / 2, self.car_width_on_map / 2
        trans = mtransforms.Affine2D().rotate_deg_around(sensor_x, sensor_y, -math.degrees(pose_local[2])).translate(
            pose_local[0] - sensor_x, pose_local[1] - sensor_y) + ax.transData
        rect.set_transform(trans)