import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import itertools

# To import carla
try:  # Make sure to install CARLA 0.8.4 (prebuilt version)
    carla_path = os.environ['CARLA_PATH']
    sys.path.insert(0, carla_path + 'PythonClient') 
except IndexError:
    pass

from carla.driving_benchmark.experiment_suites import CoRL2017  # For collisions results
from carla.driving_benchmark.metrics import Metrics  # For collisions results

# ------------------------------------------------------------------------------
# Configurations 
# ------------------------------------------------------------------------------
# directory = "/home/heraqi/scripts/int-end-to-end-ad-carla-valeo/_benchmarks_results/"
directory = "/home/heraqi/scripts/int-end-to-end-ad-carla-valeo/_benchmarks_results_papers_official/"
test_report_folder_town1 = ["/../_benchmarks_results_papers_official/AUCdata_experiment_F034BF_AUC2_data_1cam-pgm_4_epoch_22_Town01",
                            "AUCdata_experiment_F034BF_AUC2_data_1cam-pgm_4_epoch_22_Town01_Our_Route_Planner"]
test_report_folder_town2 = ["/../_benchmarks_results_papers_official/AUCdata_experiment_F034BF_AUC2_data_1cam-pgm_4_epoch_22_Town02",
                            "AUCdata_experiment_F034BF_AUC2_data_1cam-pgm_4_epoch_22_Town02_Our_Route_Planner"]
models_labels = ["1cam + LiDAR", "1cam + LiDAR + our Route Planner)"]

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
tasks_per_model_town_labels = []
for m in models_labels:
    tasks_per_model_town_labels.append(m + " - training weathers")
    tasks_per_model_town_labels.append(m + " - testing weathers")
tasks_per_model_town_labels.insert(0, "1cam (CIL paper results) - testing weathers")
tasks_per_model_town_labels.insert(0, "1cam (CIL paper results) - training weathers")

success_rate_metrics = ['episodes_fully_completed', 'episodes_completion']
nbr_experiments = 24  # 4 tasks * 6 weather conditions
nbr_paths_in_each_experiment = 25
fig = plt.figure(figsize=(15, 5))
fig.subplots_adjust(wspace=0.4)
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
        episodes_fully_completed_trainweathers_values = np.array(
            list(episodes_fully_completed_trainweathers_dict.values()))
        episodes_fully_completed_testweathers_dict = {k: episodes_fully_completed[k] for k in
                                                      experiment_suite.test_weathers}
        episodes_fully_completed_testweathers_values = np.array(
            list(episodes_fully_completed_testweathers_dict.values()))

        train_weather_straight = np.mean(100 * episodes_fully_completed_trainweathers_values[:, 0, :])  # All weathers, first task, all paths (source-destination pairs)
        train_weather_one_turn = np.mean(100 * episodes_fully_completed_trainweathers_values[:, 1, :])
        train_weather_navigation = np.mean(100 * episodes_fully_completed_trainweathers_values[:, 2, :])
        train_weather_nav_dynamic = np.mean(100 * episodes_fully_completed_trainweathers_values[:, 3, :])
        train_weather_straight_std = np.std(100 * episodes_fully_completed_trainweathers_values[:, 0, :])
        train_weather_one_turn_std = np.std(100 * episodes_fully_completed_trainweathers_values[:, 1, :])
        train_weather_navigation_std = np.std(100 * episodes_fully_completed_trainweathers_values[:, 2, :])
        train_weather_nav_dynamic_std = np.std(100 * episodes_fully_completed_trainweathers_values[:, 3, :])
        
        test_weather_straight = np.mean(100 * episodes_fully_completed_testweathers_values[:, 0, :])
        test_weather_one_turn = np.mean(100 * episodes_fully_completed_testweathers_values[:, 1, :])
        test_weather_navigation = np.mean(100 * episodes_fully_completed_testweathers_values[:, 2, :])
        test_weather_nav_dynamic = np.mean(100 * episodes_fully_completed_testweathers_values[:, 3, :])
        test_weather_straight_std = np.std(100 * episodes_fully_completed_testweathers_values[:, 0, :])
        test_weather_one_turn_std = np.std(100 * episodes_fully_completed_testweathers_values[:, 1, :])
        test_weather_navigation_std = np.std(100 * episodes_fully_completed_testweathers_values[:, 2, :])
        test_weather_nav_dynamic_std = np.std(100 * episodes_fully_completed_testweathers_values[:, 3, :])

        # Plot results
        data_to_visualize.append(
            [train_weather_straight, train_weather_one_turn, train_weather_navigation, train_weather_nav_dynamic])
        data_to_visualize_std_dev.append(
            [train_weather_straight_std, train_weather_one_turn_std, train_weather_navigation_std,
             train_weather_nav_dynamic_std])

        data_to_visualize.append(
            [test_weather_straight, test_weather_one_turn, test_weather_navigation, test_weather_nav_dynamic])
        data_to_visualize_std_dev.append(
            [test_weather_straight_std, test_weather_one_turn_std, test_weather_navigation_std,
             test_weather_nav_dynamic_std])

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

    df.plot(kind='bar', width=0.8, ax=ax, legend=False)
    # df.iloc[:, 0:6].plot(kind='bar', width=0.8, ax=ax, legend=False)  # TODO: this line is to show some models only if needed instead of all

    xticks = ['Straight', 'One Turn', 'Navigation', 'Dynamic\nNavigation']
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(xticks, rotation='vertical')
    plt.subplots_adjust(bottom=0.4)
    if town == 0:  # For legend
        leg_handles, leg_labels = ax.get_legend_handles_labels()

plt.legend(leg_handles, leg_labels, loc='center left', bbox_to_anchor=(1, 0.5))
# plt.show(block=False)
plt.savefig(directory + 'success_rates.png')

# ------------------------------------------------------------------------------
# Generate Collisions Results
# ------------------------------------------------------------------------------
ignore_state_of_the_art_comparision = True
compare_with_pretrained_model = True  # False: will use paper reported results
if compare_with_pretrained_model and not ignore_state_of_the_art_comparision:
    test_report_folder_town1.insert(0, "Pretrained_experiment_imitation_model_Town01")
    test_report_folder_town2.insert(0, "Pretrained_experiment_imitation_model_Town02")
    models_labels.insert(0, "CIL model")
if ignore_state_of_the_art_comparision:
    tasks_per_model_town_labels = []
    for m in models_labels:
        tasks_per_model_town_labels.append(m + " - training weathers")
        tasks_per_model_town_labels.append(m + " - testing weathers")

infraction_metrics = ['intersection_otherlane', 'intersection_offroad', 'collision_other', 'collision_vehicles',
                      'collision_pedestrians']
infraction_labels = ['Opposite lane', 'Sidewalk', 'Collision with\nstatic object', 'Collision with\ncar',
                     'Collision with \npedestrian']
nbr_experiments = 24  # 4 tasks * 6 weather conditions
nbr_paths_in_each_experiment = 25
fig = plt.figure(figsize=(15, 5))
fig.subplots_adjust(wspace=0.4)
for town in range(2):
    data_to_visualize = []
    data_to_visualize_std_dev = []
    if town == 0:
        print("On Town 1 (training town):")
        ax = plt.subplot(131)
        ax.title.set_text('Town 1 (training town)')
        experiment_suite = CoRL2017("Town01")
        ax.set_ylabel('Average distance (in kilometers) traveled\nbetween two infractions. Higher is better.\nMeasured on '
                      'Dymanic Navigation tasks only')
        # ax.set_ylabel('Number of infractions')
    else:
        print("On Town 2 (testing town):")
        ax = plt.subplot(132)
        ax.title.set_text('Town 2 (testing town)')
        experiment_suite = CoRL2017("Town02")
    ax.set_xlabel('Infraction')
    print("\t1cam (CIL paper)")
    if town == 0:
        print("\t\tOn same training weathers:")
        print("\t\t\t33.4, 12.9, 5.38, 3.26, 6.35")
        if not compare_with_pretrained_model and not ignore_state_of_the_art_comparision:
            data_to_visualize.append([33.4, 12.9, 5.38, 3.26, 6.35])
            data_to_visualize_std_dev.append([0, 0, 0, 0, 0])
        print("\t\tOn new testing weathers:")
        print("\t\t\t57.3, 57, 4.05, 1.86, 11.2")  # Should be >57 instead pf 57
        if not compare_with_pretrained_model and not ignore_state_of_the_art_comparision:
            data_to_visualize.append([57.3, 57, 4.05, 1.86, 11.2])  # Should be >57 instead pf 57
            data_to_visualize_std_dev.append([0, 0, 0, 0, 0])
    else:
        print("\t\tOn same training weathers:")
        print("\t\t\t1.12, 0.76, 0.40, 0.59, 1.88")
        if not compare_with_pretrained_model and not ignore_state_of_the_art_comparision:
            data_to_visualize.append([1.12, 0.76, 0.40, 0.59, 1.88])
            data_to_visualize_std_dev.append([0, 0, 0, 0, 0])
        print("\t\tOn new testing weathers:")
        print("\t\t\t0.78, 0.81, 0.28, 0.44, 1.41")
        if not compare_with_pretrained_model and not ignore_state_of_the_art_comparision:
            data_to_visualize.append([0.78, 0.81, 0.28, 0.44, 1.41])
            data_to_visualize_std_dev.append([0, 0, 0, 0, 0])
    town_folders = test_report_folder_town1 if (town == 0) else test_report_folder_town2
    for f in range(len(town_folders)):  # For each model for that town
        metrics_object = Metrics(experiment_suite.metrics_parameters, experiment_suite.dynamic_tasks)
        metrics = metrics_object.compute(directory + town_folders[f])

        # Store driven_kilometers
        driven_kilometers_metrics = metrics['driven_kilometers']
        driven_kilometers_metrics_trainweathers_dict = {k: driven_kilometers_metrics[k] for k in
                                                        experiment_suite.train_weathers}
        driven_kilometers_metrics_trainweathers_values = list(driven_kilometers_metrics_trainweathers_dict.values())
        driven_kilometers_metrics_testweathers_dict = {k: driven_kilometers_metrics[k] for k in
                                                       experiment_suite.test_weathers}
        driven_kilometers_metrics_testweathers_values = list(driven_kilometers_metrics_testweathers_dict.values())

        # Retrieve infractions
        train_infraction_mean = np.zeros(5)
        train_infraction_std = np.zeros(5)
        test_infraction_mean = np.zeros(5)
        test_infraction_std = np.zeros(5)
        for (inf, infraction_metric) in enumerate(infraction_metrics):
            metric_data = metrics[infraction_metric]
            # Fix if number of paths shorter than 25 in some cases
            for key in metric_data:
                for indx in range(len(metric_data[key])):
                    if len(metric_data[key][indx]) < 25:
                        metric_data[key][indx] = metric_data[key][indx] + [0]*(25 - len(metric_data[key][indx]))
            metric_data_trainweathers_dict = {k: metric_data[k] for k in experiment_suite.train_weathers}
            metric_data_trainweathers_values = list(metric_data_trainweathers_dict.values())
            metric_data_testweathers_dict = {k: metric_data[k] for k in experiment_suite.test_weathers}
            metric_data_testweathers_values = list(metric_data_testweathers_dict.values())

            # Select infraction all weathers, Dynamic Navigation task only (task index 3), and all paths
            infractions = np.array(metric_data_trainweathers_values)[:, 3, :]
            train_infraction_mean[inf] = np.sum(np.array(driven_kilometers_metrics_trainweathers_values)[:, 3]) / (np.sum(infractions)+1)
            # train_infraction_mean[inf] = np.sum(infractions)
            train_infraction_std[inf] = 0
            infractions = np.array(metric_data_testweathers_values)[:, 3, :]
            test_infraction_mean[inf] = np.sum(np.array(driven_kilometers_metrics_testweathers_values)[:, 3]) / (np.sum(infractions)+1)
            # test_infraction_mean[inf] = np.sum(infractions)
            test_infraction_std[inf] = 0

        # Plot results
        data_to_visualize.append([train_infraction_mean[0], train_infraction_mean[1], train_infraction_mean[2],
                                  train_infraction_mean[3], train_infraction_mean[4]])
        data_to_visualize_std_dev.append([train_infraction_std[0], train_infraction_std[1], train_infraction_std[2],
                                          train_infraction_std[3], train_infraction_std[4]])
        data_to_visualize.append([test_infraction_mean[0], test_infraction_mean[1], test_infraction_mean[2],
                                  test_infraction_mean[3], test_infraction_mean[4]])
        data_to_visualize_std_dev.append([test_infraction_std[0], test_infraction_std[1], test_infraction_std[2],
                                          test_infraction_std[3], test_infraction_std[4]])

        # Print results
        print("\t" + models_labels[f])
        print("\t\tOn same training weathers:")
        print("\t\t\t" + "{0:.2f}".format(train_infraction_mean[0]) + " " +
              "{0:.2f}".format(train_infraction_mean[1]) + " " +
              "{0:.2f}".format(train_infraction_mean[2]) + " " +
              "{0:.2f}".format(train_infraction_mean[3]) + " " +
              "{0:.2f}".format(train_infraction_mean[4]))
        print("\t\tOn new testing weathers:")
        print("\t\t\t" + "{0:.2f}".format(test_infraction_mean[0]) + " " +
              "{0:.2f}".format(test_infraction_mean[1]) + " " +
              "{0:.2f}".format(test_infraction_mean[2]) + " " +
              "{0:.2f}".format(test_infraction_mean[3]) + " " +
              "{0:.2f}".format(test_infraction_mean[4]))

    # Fix xticks
    df = pd.DataFrame(index=[1, 2, 3, 4, 5], columns=tasks_per_model_town_labels, data=np.array(data_to_visualize).T)
    # df_yerr = pd.DataFrame(index=[1,2,3,4], columns=labels, data=np.array(data_to_visualize_std_dev).T)
    # df.plot(kind='bar',yerr=df_yerr, ax=ax, legend=False)

    df.plot(kind='bar', width=0.8, ax=ax, legend=False)
    # df.iloc[:, 0:6].plot(kind='bar', width=0.8, ax=ax, legend=False)  # TODO: this line is to show some models only if needed instead of all

    xticks = infraction_labels
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(xticks, rotation='vertical')
    plt.subplots_adjust(bottom=0.4)
    if town == 0:  # For legend
        leg_handles, leg_labels = ax.get_legend_handles_labels()

plt.legend(leg_handles, leg_labels, loc='center left', bbox_to_anchor=(1, 0.5))
# plt.show(block=False)
plt.savefig(directory + 'collisions.png')

