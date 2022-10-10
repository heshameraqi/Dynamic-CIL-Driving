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

from generate_latex import generate_latex_table

# ------------------------------------------------------------------------------
# Configurations
# ------------------------------------------------------------------------------
directory = "/home/heraqi/scripts/int-end-to-end-ad-carla-valeo/_official_paper_results_tables/"  # to save result figures & LaTeX tables
models_labels = ["Camera (CIL paper results)", "Camera (CIL paper pretrained model)", "Camera (our data)", "Camera + LiDAR (our data)", "Camera + LiDAR + Our Route Planner (our data)"]
infraction_labels = ['Collision to\na Pedestrian', 'Collision to\na Vehicle', 'Collision to a\nStatic Object', 'Going Outside\nof Road', 'Invading the\nOpposite Lane']

# ------------------------------------------------------------------------------
# Generate Success Rate Results
# ------------------------------------------------------------------------------
data_to_visualize_town1 = [
    # Paper
        [95, 89, 86, 83],  # train weathers
        [98, 90, 84, 82],  # test weathers
    # Pre-trained  model
        [99, 88, 78, 80],  # train weathers
        [100, 94, 84, 74],  # test weathers
    # 1 Camera
        [100, 97, 87, 84],  # train weathers
        [100, 98, 88, 82],  # test weathers
    # 1 Camera + PGM
        [100, 100, 92, 86],  # train weathers
        [100, 100, 92, 86],   # test weathers
    # 1 Camera + PGM + New Route Planner
        [100, 100, 96, 94],  # train weathers
        [100, 100, 96, 96]   # test weathers
]

data_to_visualize_town2 = [
    # Paper
        [97, 59, 40, 38],  # train weathers
        [80, 48, 44, 42],  # test weathers
    # Pre-trained  model
        [89, 56, 35, 28],  # train weathers
        [92, 74, 58, 54],  # test weathers
    # 1 Camera
        [99, 57, 33, 26],  # train weathers
        [100, 72, 34, 30],  # test weathers
    # 1 Camera + PGM
        [100, 92, 68, 53],  # train weathers
        [100, 92, 68, 64],   # test weathers
    # 1 Camera + PGM + New Route Planner
        [100, 92, 100, 89],  # train weathers
        [100, 92, 100, 88]   # test weathers

]
tasks_per_model_town_labels = []
for m in models_labels:
    tasks_per_model_town_labels.append(m + ", training weathers")
    tasks_per_model_town_labels.append(m + ", testing weathers")
fig = plt.figure(figsize=(15, 5))
fig.subplots_adjust(wspace=0.4)
for town in range(2):
    if town == 0:
        data_to_visualize = data_to_visualize_town1
        print("On Town 1 (training town):")
        ax = plt.subplot(131)
        ax.title.set_text('Town 1 (training town)')
        experiment_suite = CoRL2017("Town01")
    else:
        data_to_visualize = data_to_visualize_town2
        print("On Town 2 (testing town):")
        ax = plt.subplot(132)
        ax.title.set_text('Town 2 (testing town)')
        experiment_suite = CoRL2017("Town02")
    ax.set_xlabel('Task')
    ax.set_ylabel('Success Rate')

    # Fix xticks
    df = pd.DataFrame(index=[1, 2, 3, 4], columns=tasks_per_model_town_labels, data=np.array(data_to_visualize).T)
    df.plot(kind='bar', width=0.8, ax=ax, legend=False)
    xticks = ['Straight', 'One Turn', 'Navigation', 'Dynamic\nNavigation']
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(xticks, rotation='vertical')
    plt.subplots_adjust(bottom=0.4)
    if town == 0:  # For legend
        leg_handles, leg_labels = ax.get_legend_handles_labels()

plt.legend(leg_handles, leg_labels, loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig(directory + 'success_rates.png')
generate_latex_table("success_rates_table.tex", data_to_visualize_town1, data_to_visualize_town2, directory)
data_to_visualize_town1_success = data_to_visualize_town1
data_to_visualize_town2_success = data_to_visualize_town2

# ------------------------------------------------------------------------------
# Generate Percentage of Distance to Goal Travelled (Get the from _gen_results_for_model_official_avgdist_collisions_from_benchmark_results.py)
# ------------------------------------------------------------------------------
data_to_visualize_town1 = [
    # Pre-trained  model
        [0.9723715870997012*100, 0.8271343154039992*100, 0.8861487765143035*100, 0.8835760047410889*100],  # train weathers
        [0.9837525161427734*100, 0.8548617833813217*100, 0.893358284934004*100, 0.8152507171250801*100],  # test weathers
    # 1 Camera
        [0.9837334596389824*100, 0.9733428495710389*100, 0.911250424694684*100, 0.9102969606785964*100],  # train weathers
        [0.9833124033320839*100, 0.9772420265279818*100, 0.9246334032715979*100, 0.8733590815859*100],  # test weathers
    # 1 Camera + PGM
        [0.9832378424570476*100, 0.984677966705396*100, 0.9270294249500062*100, 0.9301743797694768*100],  # train weathers
        [0.9831625022835471*100, 0.9846489784308439*100, 0.9271347791686965*100, 0.9289122607639186*100],   # test weathers
    # 1 Camera + PGM + Our Route Planner
        [0.9835514412609199*100, 0.9844013861977728*100, 0.9602073072188707*100, 0.981644656912759*100],  # train weathers
        [0.983636368849259*100, 0.9843732112084461*100, 0.9603676313138593*100, 0.9858938126900142*100]   # test weathers 
]

data_to_visualize_town2 = [
    # Pre-trained  model
        [0.903808623119825*100, 0.5481025492217879*100, 0.09681466820940944*100, 0.1735490057236999*100],  # train weathers
        [0.9269880847060882*100, 0.6060171319921884*100, 0.4537314676745482*100, 0.35125365142695814*100],  # test weathers
    # 1 Camera
        [0.9573838668069797*100, 0.5608602430934263*100, 0.16923308514468594*100, 0.09526710870886865*100],  # train weathers
        [0.9665196931743139*100, 0.6717833433294143*100, 0.1693271436455083*100, 0.2941361055997687*100],  # test weathers
    # 1 Camera + PGM
        [0.9667580588411325*100, 0.9004844355747251*100, 0.7695419801866725*100, 0.37507709112554094*100],  # train weathers
        [0.9668055706435958*100, 0.9151916787478492*100, 0.7685535109311126*100, 0.5990014489783243*100],   # test weathers
    # 1 Camera + PGM + Our Route Planner
        [0.9663748562272402*100, 0.8816823603114143*100, 0.9829955967027924*100, 0.8088348606098378*100],  # train weathers
        [0.9663748562272402*100, 0.8816823603114143*100, 0.9829955967027924*100, 0.7780473744581162*100]   # test weathers   
]

tasks_per_model_town_labels = []
for m in models_labels:
    tasks_per_model_town_labels.append(m + " - training weathers")
    tasks_per_model_town_labels.append(m + " - testing weathers")
fig = plt.figure(figsize=(15, 5))
fig.subplots_adjust(wspace=0.4)
for town in range(2):
    if town == 0:
        data_to_visualize = data_to_visualize_town1
        print("On Town 1 (training town):")
        ax = plt.subplot(131)
        ax.title.set_text('Town 1 (training town)')
        experiment_suite = CoRL2017("Town01")
    else:
        data_to_visualize = data_to_visualize_town2
        print("On Town 2 (testing town):")
        ax = plt.subplot(132)
        ax.title.set_text('Town 2 (testing town)')
        experiment_suite = CoRL2017("Town02")
    ax.set_xlabel('Task')
    ax.set_ylabel('Average Percentage of Distance to Goal Travelled')

    # Fix xticks
    # [2:] because CIL paper results doesn't include this info
    df = pd.DataFrame(index=[1, 2, 3, 4], columns=tasks_per_model_town_labels[2:], data=np.array(data_to_visualize).T)
    df.plot(kind='bar', width=0.8, ax=ax, legend=False)
    xticks = ['Straight', 'One Turn', 'Navigation', 'Dynamic\nNavigation']
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(xticks, rotation='vertical')
    plt.subplots_adjust(bottom=0.4)
    if town == 0:  # For legend
        leg_handles, leg_labels = ax.get_legend_handles_labels()

plt.legend(leg_handles, leg_labels, loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig(directory + 'average_distance_percentage.png')
generate_latex_table("average_distance_percentage_table.tex", data_to_visualize_town1, data_to_visualize_town2, directory)
generate_latex_table("success_rates_and_avg_dist_table.tex", data_to_visualize_town1, data_to_visualize_town2, directory,
                     data_to_visualize_town1_success, data_to_visualize_town2_success)

# ------------------------------------------------------------------------------
# Generate Collisions Results (Get the from _gen_results_for_model_official_avgdist_collisions_from_benchmark_results.py)
# ------------------------------------------------------------------------------
data_to_visualize_town1 = [
    # Paper
        [33.4, 12.9, 5.38, 3.26, 6.35],  # train weathers
        [57.3, 57, 4.05, 1.86, 11.2],  # test weathers
    # Pre-trained model
        [7.1498341977979845, 1.3490253203392424, 5.499872459844604, 14.299668395595969, 4.766556131865324],  # train weathers
        [19.47678969340116, 0.8853086224273254, 5.564797055257474, 12.98452646226744, 9.73839484670058],  # test weathers
    # 1 Camera
        [6.084057541801888, 1.5934436419004945, 5.148048689216982, 11.154105493303462, 13.384926591964154],  # train weathers
        [5.3166293374753115, 1.3291573343688279, 2.6583146687376558, 7.974944006212967, 15.949888012425934],  # test weathers
    # 1 Camera + PGM
        [60.95687023301055, 1.128830930240936, 3.0478435116505276, 10.159478372168424, 8.708124319001508],  # train weathers
        [15.75540158212851, 1.0865794194571385, 3.151080316425702, 10.50360105475234, 10.50360105475234],   # test weathers
    # 1 Camera + PGM + Our Route Planner
        [32.021318169973966, 0.9558602438798199, 1.9406859496953919, 64.04263633994793, 21.34754544664931],  # train weathers
        [10.790151656372812, 1.198905739596979, 2.697537914093203, 32.370454969118434, 32.370454969118434]   # test weathers
]

data_to_visualize_town2 = [
    # Paper
        [1.12, 0.76, 0.40, 0.59, 1.88],  # train weathers
        [0.78, 0.81, 0.28, 0.44, 1.41],  # test weathers
    # Pre-trained  mode
        [0.9880644977067669, 0.17817556516023664, 0.2898322526606516, 0.45286289478226815, 0.5055213709197411],  # train weathers
        [2.1769452594808967, 0.17316610018598044, 0.8465898231314599, 0.896389224492134, 1.6931796462629198],  # test weathers
    # 1 Camera
        [1.4934906731581685, 0.43926196269357903, 0.2574983919238222, 0.5895357920361192, 0.7724951757714665],  # train weathers
        [3.1423437864315416, 0.6982986192070092, 0.3307730301506886, 0.7855859466078854, 0.8978125104090119],  # test weathers
    # 1 Camera + PGM
        [25.006460902997603, 0.5103359367958694, 0.5436187152825566, 0.9617869578076002, 0.9261652186295408],  # train weathers
        [12.59740713047995, 0.8398271420319967, 0.8998147950342822, 1.5746758913099939, 1.0497839275399958],   # test weathers
    # 1 Camera + PGM + Our Route Planner
        [5.501982073060556, 1.8339940243535184, 1.0316216386988541, 2.750991036530278, 11.003964146121112],  # train weathers
        [16.145379161387964, 2.0181723951734956, 1.2419522431836896, 3.229075832277593, 16.145379161387964]   # test weathers
]

fig = plt.figure(figsize=(15, 5))
fig.subplots_adjust(wspace=0.4)
for town in range(2):
    if town == 0:
        data_to_visualize = data_to_visualize_town1
        print("On Town 1 (training town):")
        ax = plt.subplot(131)
        ax.title.set_text('Town 1 (training town)')
        experiment_suite = CoRL2017("Town01")
        ax.set_ylabel('Average distance (in kilometers) traveled\nbetween two infractions. Higher is better.\nMeasured on '
                      'Dymanic Navigation tasks only')
        # ax.set_ylabel('Number of infractions')
    else:
        data_to_visualize = data_to_visualize_town2
        print("On Town 2 (testing town):")
        ax = plt.subplot(132)
        ax.title.set_text('Town 2 (testing town)')
        experiment_suite = CoRL2017("Town02")
    ax.set_xlabel('Infraction')

    # Fix xticks
    # [2:] to ignore paper resutls because it seems much higher than their pretrained model
    df = pd.DataFrame(index=[1, 2, 3, 4, 5], columns=tasks_per_model_town_labels[2:], data=np.array(data_to_visualize[2:]).T)
    df.plot(kind='bar', width=0.8, ax=ax, legend=False)
    xticks = infraction_labels
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(xticks, rotation='vertical')
    plt.subplots_adjust(bottom=0.4)
    if town == 0:  # For legend
        leg_handles, leg_labels = ax.get_legend_handles_labels()

plt.legend(leg_handles, leg_labels, loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig(directory + 'collisions.png')
generate_latex_table("collisions_table.tex", data_to_visualize_town1, data_to_visualize_town2, directory)
