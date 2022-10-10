# To import carla
import os
import sys
try:  # Make sure to install CARLA 0.8.4 (prebuilt version)
    carla_path = os.environ['CARLA_PATH']
    sys.path.insert(0, carla_path + 'PythonClient')
except IndexError:
    pass
    
from carla.driving_benchmark.metrics import Metrics
# from carla.driving_benchmark import results_printer
import results_printer

# ------------------------------------------------------------------------------------------
# Model results directory
# ------------------------------------------------------------------------------------------
# path = "/home/heraqi/scripts/int-end-to-end-ad-carla-valeo/_benchmarks_papers_official_results/AUCdata_experiment_F034BF_AUC2_data_1cam-pgm_4_epoch_22_Town01_Our_Route_Planner"
# path = "/home/heraqi/scripts/int-end-to-end-ad-carla-valeo/_benchmarks_papers_official_results/AUCdata_experiment_F034BF_AUC2_data_1cam-pgm_4_epoch_22_Town01"
path = r"C:\Work\Software\int-end-to-end-ad-carla-valeo\_benchmarks_work_zones_papers_official_results\AUCdata_WZAlgo_WithoutRect_F034BF_AUC2_data_1cam-pgm_4_epoch_22_Town02"
# ------------------------------------------------------------------------------------------

# These parameters won't matter
work_zones_benchmark = True
model_inputs_mode = "1cam-pgm"
our_experiment_sensor_setup = True
town = 'Town01'
if work_zones_benchmark:
    from experiment_working_zones import experiment
else:
    from experiment_corl17 import experiment
experiment = experiment(model_inputs_mode, our_experiment_sensor_setup, town)

# Printing
metrics_object = Metrics(experiment.metrics_parameters, experiment.dynamic_tasks)
benchmark_summary = metrics_object.compute(path)
print("")
print("")
print("----- Printing results for training weathers (Seen in Training) -----")
print("")
print("")
results_printer.print_summary(benchmark_summary, experiment.train_weathers, path)
print("")
print("")
print("----- Printing results for test weathers (Unseen in Training) -----")
print("")
print("")
results_printer.print_summary(benchmark_summary, experiment.test_weathers, path)