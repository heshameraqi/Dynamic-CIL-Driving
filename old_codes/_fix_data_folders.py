import os
import pdb
import shutil

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

'''
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

dir = "/home/heraqi/data/int-end-to-end-ad/auc.carla.dataset_00/train"
episodes = [episode for episode in os.listdir(dir)]
for i in range(len(episodes)):
    if len([f for f in os.listdir(os.path.join(dir, str(i)) + '/CameraDepth')]) < 9000:
        shutil.rmtree(os.path.join(dir, str(i)))
        for j in range(i, len(episodes)):
            shutil.move(os.path.join(dir, str(j+1)), os.path.join(dir, str(j)))
        episodes = [episode for episode in os.listdir(dir)]
    



