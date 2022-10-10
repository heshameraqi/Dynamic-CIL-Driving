import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# To import carla
try:  # Make sure to install CARLA 0.8.2 (prebuilt version)
    carla_path = os.environ['CARLA_PATH']
    sys.path.insert(0, carla_path + 'PythonClient')
except IndexError:
    pass

from carla.driving_benchmark.experiment_suites import CoRL2017  # For collisions results
from carla.driving_benchmark.metrics import Metrics  # For collisions results

# ------------------------------------------------------------------------------
# Configurations
# ------------------------------------------------------------------------------
directory = "/home/heraqi/scripts/int-end-to-end-ad-carla-valeo/_benchmarks_results/"
test_report_folder_town1 = ["CIL_data_experiment_Town01_1cam",
                            # CILdata_Town01_normalthreshold CILdata_Town01_lowerthreshold CIL_data_experiment_Town01_1cam
                            "AUCdata_experiment_Town01_1cam-1",
                            "AUCdata_experiment_Town01_1cam-1",
                            "AUCdata_experiment_Town01_1cam-1"]
test_report_folder_town2 = ["CIL_data_experiment_Town02_1cam-47",
                            # CILdata_Town02_normalthreshold CILdata_Town02_lowerthreshold CIL_data_experiment_Town02_1cam-47
                            "AUCdata_experiment_Town02_1cam-120",
                            "AUCdata_experiment_Town02_1cam-120",
                            "AUCdata_experiment_Town02_1cam-120"]

models_labels = ["1cam (CIL data)", "1cam (our data)", "3cams (our data)", "3cams+LiDAR (our data)"]

tasks_per_model_town_labels = ["1cam (CIL paper) - training weathers",
                               "1cam (CIL paper) - testing weathers",
                               "1cam (CIL data) - training weathers",
                               "1cam (CIL data) - testing weathers",
                               "1cam (our data) - training weathers",
                               "1cam (our data) - testing weathers",
                               "3cams (our data) - training weathers",
                               "3cams (our data) - testing weathers",
                               "3cams+LiDAR (our data) - training weathers",
                               "3cams+LiDAR (our data) - testing weathers"
                               ]

'''
Data is collected according to CARLA benchmark (Alexey Dosovitskiy, "CARLA: An Open Urban Driving Simulator") only 
training weather conditions ([1:ClearNoon, 3:WetNoon, 6:HardRainNoon, 8:ClearSunset], test were [4:WetCloudyNoon, 14:SoftRainSunset])
'''
weathers_dict = {
    0: "Default",
    1: "ClearNoon - Train",  # Train
    2: "CloudyNoon",
    3: "WetNoon - Train",  # Train
    4: "WetCloudyNoon - Test",  # Test
    5: "MidRainyNoon",
    6: "HardRainNoon - Train",  # Train
    7: "SoftRainNoon",
    8: "ClearSunset - Train",  # Train
    9: "CloudySunset",
    10: "WetSunset",
    11: "WetCloudySunset",
    12: "MidRainSunset",
    13: "HardRainSunset",
    14: "SoftRainSunset - Test",  # Test
}

# ------------------------------------------------------------------------------
# Generate Success Rate Results
# ------------------------------------------------------------------------------
metrics_to_average = ['episodes_fully_completed', 'episodes_completion']
infraction_metrics = ['collision_pedestrians', 'collision_vehicles', 'collision_other', 'intersection_offroad',
                      'intersection_otherlane']
nbr_experiments = 24  # 4 tasks * 6 weather conditions
nbr_paths_in_each_experiment = 25
fig = plt.figure(figsize=(13,5))
for town in range(2):
    data_to_visualize = []
    data_to_visualize_std_dev = []
    if town == 0:
        print("On Town 1 (training town):")
        ax = plt.subplot(131)
        ax.title.set_text('Town 1 (training town)')
        experiment_suite = CoRL2017("Town01")
    else:
        print("On Town 2 (testing town):")
        ax = plt.subplot(132)
        ax.title.set_text('Town 2 (testing town)')
        experiment_suite = CoRL2017("Town02")
    ax.set_xlabel('Task')
    ax.set_ylabel('Success Rate')
    print("\t1cam (CIL paper)")
    if town == 0:
        print("\t\tOn same training weathers:")
        print("\t\t\t95.00, 89.00, 86.00, 83.00")
        data_to_visualize.append([95, 89, 86, 83])
        data_to_visualize_std_dev.append([0, 0, 0, 0])
        print("\t\tOn new testing weathers:")
        print("\t\t\t98.00, 90.00, 84.00, 82.00")
        data_to_visualize.append([98, 90, 84, 82])
        data_to_visualize_std_dev.append([0, 0, 0, 0])
    else:
        print("\t\tOn same training weathers:")
        print("\t\t\t97.00, 59.00, 40.00, 38.00")
        data_to_visualize.append([97, 59, 40, 38])
        data_to_visualize_std_dev.append([0, 0, 0, 0])
        print("\t\tOn new testing weathers:")
        print("\t\t\t80.00, 48.00, 44.00, 42.00")
        data_to_visualize.append([80, 48, 44, 42])
        data_to_visualize_std_dev.append([0, 0, 0, 0])
    town_folders = test_report_folder_town1 if (town == 0) else test_report_folder_town2
    for f in range(len(town_folders)):
        metrics_object = Metrics(experiment_suite.metrics_parameters, experiment_suite.dynamic_tasks)
        metrics = metrics_object.compute(directory + town_folders[f])

        episodes_fully_completed = metrics["episodes_fully_completed"]
        episodes_fully_completed_trainweathers_dict = {k: episodes_fully_completed[k] for k in
                                                       experiment_suite.train_weathers}
        episodes_fully_completed_trainweathers_values = np.array(list(episodes_fully_completed_trainweathers_dict.values()))
        episodes_fully_completed_testweathers_dict = {k: episodes_fully_completed[k] for k in
                                                      experiment_suite.test_weathers}
        episodes_fully_completed_testweathers_values = np.array(list(episodes_fully_completed_testweathers_dict.values()))

        train_weather_straight = np.mean(100*episodes_fully_completed_trainweathers_values[:, 0, :])
        train_weather_one_turn = np.mean(100*episodes_fully_completed_trainweathers_values[:, 1, :])
        train_weather_navigation = np.mean(100*episodes_fully_completed_trainweathers_values[:, 2, :])
        train_weather_nav_dynamic = np.mean(100*episodes_fully_completed_trainweathers_values[:, 3, :])
        train_weather_straight_std = np.std(100*episodes_fully_completed_trainweathers_values[:, 0, :])
        train_weather_one_turn_std = np.std(100*episodes_fully_completed_trainweathers_values[:, 1, :])
        train_weather_navigation_std = np.std(100*episodes_fully_completed_trainweathers_values[:, 2, :])
        train_weather_nav_dynamic_std = np.std(100*episodes_fully_completed_trainweathers_values[:, 3, :])

        test_weather_straight = np.mean(100 * episodes_fully_completed_testweathers_values[:, 0, :])
        test_weather_one_turn = np.mean(100 * episodes_fully_completed_testweathers_values[:, 1, :])
        test_weather_navigation = np.mean(100 * episodes_fully_completed_testweathers_values[:, 2, :])
        test_weather_nav_dynamic = np.mean(100 * episodes_fully_completed_testweathers_values[:, 3, :])
        test_weather_straight_std = np.std(100 * episodes_fully_completed_testweathers_values[:, 0, :])
        test_weather_one_turn_std = np.std(100 * episodes_fully_completed_testweathers_values[:, 1, :])
        test_weather_navigation_std = np.std(100 * episodes_fully_completed_testweathers_values[:, 2, :])
        test_weather_nav_dynamic_std = np.std(100 * episodes_fully_completed_testweathers_values[:, 3, :])

        # Plot results
        data_to_visualize.append([train_weather_straight, train_weather_one_turn, train_weather_navigation, train_weather_nav_dynamic])
        data_to_visualize_std_dev.append([train_weather_straight_std, train_weather_one_turn_std, train_weather_navigation_std, train_weather_nav_dynamic_std])

        data_to_visualize.append([test_weather_straight, test_weather_one_turn, test_weather_navigation, test_weather_nav_dynamic])
        data_to_visualize_std_dev.append([test_weather_straight_std, test_weather_one_turn_std, test_weather_navigation_std, test_weather_nav_dynamic_std])

        # Print results
        print("\t" + models_labels[f])
        print("\t\tOn same training weathers:")
        '''print("\t\t\t" + "{0:.2f}".format(train_weather_straight) + "±" + "{0:.2f}".format(train_weather_straight_std) + " " + 
              "{0:.2f}".format(train_weather_one_turn) + "±" + "{0:.2f}".format(train_weather_one_turn_std) + " " + 
              "{0:.2f}".format(train_weather_navigation) + "±" + "{0:.2f}".format(train_weather_navigation_std) + " " + 
              "{0:.2f}".format(train_weather_nav_dynamic) + "±" + "{0:.2f}".format(train_weather_nav_dynamic_std))'''
        print("\t\t\t" + "{0:.2f}".format(train_weather_straight) + " " +
              "{0:.2f}".format(train_weather_one_turn) + " " +
              "{0:.2f}".format(train_weather_navigation) + " " +
              "{0:.2f}".format(train_weather_nav_dynamic))
        print("\t\tOn new testing weathers:")
        '''print("\t\t\t" + "{0:.2f}".format(test_weather_straight) + "±" + "{0:.2f}".format(test_weather_straight_std) + " " + 
              "{0:.2f}".format(test_weather_one_turn) + "±" + "{0:.2f}".format(test_weather_one_turn_std) + " " + 
              "{0:.2f}".format(test_weather_navigation) + "±" + "{0:.2f}".format(test_weather_navigation_std) + " " + 
              "{0:.2f}".format(test_weather_nav_dynamic) + "±" + "{0:.2f}".format(test_weather_nav_dynamic_std))'''
        print("\t\t\t" + "{0:.2f}".format(test_weather_straight) + " " +
              "{0:.2f}".format(test_weather_one_turn) + " " +
              "{0:.2f}".format(test_weather_navigation) + " " +
              "{0:.2f}".format(test_weather_nav_dynamic))

    # Fix xticks
    df = pd.DataFrame(index=[1, 2, 3, 4], columns=tasks_per_model_town_labels, data=np.array(data_to_visualize).T)
    # df_yerr = pd.DataFrame(index=[1,2,3,4], columns=labels, data=np.array(data_to_visualize_std_dev).T)
    # df.plot(kind='bar',yerr=df_yerr, ax=ax, legend=False)

    # df.plot(kind='bar', width=0.8, ax=ax, legend=False)
    df.iloc[:, 0:6].plot(kind='bar', width=0.8, ax=ax, legend=False)  # TODO: take previous line instead

    xticks = ['Straight','One Turn','Navigation','Dynamic Navigation']
    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels(xticks, rotation='vertical')
    plt.subplots_adjust(bottom=0.4)
    if town == 0:  # For legend
        leg_handles, leg_labels = ax.get_legend_handles_labels()

plt.legend(leg_handles, leg_labels, loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


    