import os
import pdb
import shutil
import pdb

'''
dirs = []
dirs.append('/home/heraqi/data/int-end-to-end-ad/auc.carla.dataset_00_1/train')
dirs.append('/home/heraqi/data/int-end-to-end-ad/auc.carla.dataset_00_2/train')
dirs.append('/home/heraqi/data/int-end-to-end-ad/auc.carla.dataset_00_3/train')
dirs.append('/home/heraqi/data/int-end-to-end-ad/auc.carla.dataset_00_4/train')
dirs.append('/home/heraqi/data/int-end-to-end-ad/auc.carla.dataset_00_5/train')
dirs.append('/home/heraqi/data/int-end-to-end-ad/auc.carla.dataset_00_6/train')
dirs.append('/home/heraqi/data/int-end-to-end-ad/auc.carla.dataset_00_7/train')
dirs.append('/home/heraqi/data/int-end-to-end-ad/auc.carla.dataset_00_8/train')
dirs.append('/home/heraqi/data/int-end-to-end-ad/auc.carla.dataset_00_9/train')

episodes_folders_count = 0
current_count = 120
for dir in dirs:
    # pdb.set_trace()
    episodes = [episode for episode in os.listdir(dir)]
    episodes_folders_count += len(episodes)

    # delete last episode if not complete
    if  len([f for f in os.listdir(os.path.join(dir, episodes[-1]) + '/CameraDepth')]) < 9000:
        print('Deleted incomplete episode folder: ' + os.path.join(dir, episodes[-1]))
        shutil.rmtree(os.path.join(dir, episodes[-1]))
        episodes_folders_count -= 1
        
    # move data
    out_dir = "/home/heraqi/data/int-end-to-end-ad/auc.carla.dataset_00/train"
    for episode in os.listdir(dir):
        shutil.move(os.path.join(dir, str(episode)), os.path.join(out_dir, str(current_count)))
        current_count += 1
        
# print('Number of episodes: ' + str(episodes_folders_count))
'''

'''dir = "/home/heraqi/data/int-end-to-end-ad/auc.carla.dataset_00/train"
episodes = [episode for episode in os.listdir(dir)]
for i in range(len(episodes)):
    if len([f for f in os.listdir(os.path.join(dir, str(i)) + '/CameraDepth')]) < 9000:
        shutil.rmtree(os.path.join(dir, str(i)))
        for j in range(i, len(episodes)):
            shutil.move(os.path.join(dir, str(j+1)), os.path.join(dir, str(j)))
        episodes = [episode for episode in os.listdir(dir)]'''
        
      
'''folder = "/mnt/sdb1/heraqi/data/int-end-to-end-ad/auc.carla.dataset_02"
datasets = [set for set in os.listdir(folder)]
outdir = "/mnt/sdb1/heraqi/data/int-end-to-end-ad/auc.carla.dataset_01/train"
episode_number = 764
for d in datasets:
    dataset_dir = os.path.join(folder, d)
    episodes = [episode for episode in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir,episode))]
    for ep in episodes:
        shutil.move(os.path.join(dataset_dir, ep), os.path.join(outdir, 'episode_' + str(episode_number).zfill(5)))
        episode_number += 1'''
    
    
'''dataset_dir = "/mnt/sdb1/heraqi/data/int-end-to-end-ad/auc.carla.dataset_01/train"
out_dataset_dir = "/mnt/sdb1/heraqi/data/int-end-to-end-ad/auc.carla.dataset_01/val"
episodes = [eps for eps in os.listdir(dataset_dir)]
episode_number = 1
out_number = 0
for eps in episodes:
    if episode_number > 550:
        shutil.move(os.path.join(dataset_dir, eps), os.path.join(out_dataset_dir, 'episode_' + str(out_number).zfill(5)))
        out_number += 1
    episode_number += 1'''

# Collect data from multiple episodes
datasets_dir = "/media/heraqi/data/heraqi/int-end-to-end-ad/auc.carla.dataset_00"
outdir = "/media/heraqi/data/heraqi/int-end-to-end-ad/auc.carla.dataset_01/train"
outdir_val = "/media/heraqi/data/heraqi/int-end-to-end-ad/auc.carla.dataset_01/val"
datasets = [set for set in os.listdir(datasets_dir) if os.path.isdir(os.path.join(datasets_dir,set))]
episode_number = 0
for d in datasets:
    dataset_dir = os.path.join(datasets_dir, d)
    episodes = [eps for eps in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir,eps))]
    for ep in episodes[:-1]:  # skip last episode because if the process was killed, that episode is incomplete TODO: episodes should be sorted alphabetically before
        shutil.move(os.path.join(dataset_dir, ep), os.path.join(outdir, 'episode_' + str(episode_number).zfill(5)))
        episode_number += 1
# Create validation dataset
all_episodes = [eps for eps in os.listdir(outdir) if os.path.isdir(os.path.join(outdir,eps))]
val_count = round(len(all_episodes)/15)
episode_number = 0
for ep in all_episodes[-val_count:]:
    shutil.move(os.path.join(outdir, ep), os.path.join(outdir_val, 'episode_' + str(episode_number).zfill(5)))
    episode_number += 1
# Move json file
first_dataset = os.path.join(datasets_dir, datasets[0])
first_dataset_json = os.path.join(first_dataset, "metadata.json")
shutil.copy(first_dataset_json, os.path.join(outdir))
shutil.copy(first_dataset_json, os.path.join(outdir_val))

# Megre new data with old
'''old_data = "/media/heraqi/data/heraqi/int-end-to-end-ad/auc.carla.dataset_00/train"
new_data = "/media/heraqi/data/heraqi/int-end-to-end-ad/auc.carla.dataset_01/train1"
start_from = 406
new_folders = [set for set in os.listdir(new_data)]
for f in new_folders:
    shutil.move(os.path.join(new_data, f), os.path.join(old_data, 'episode_' + str(start_from).zfill(5)))
    start_from += 1'''