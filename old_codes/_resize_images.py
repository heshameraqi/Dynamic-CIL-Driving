import scipy.misc
import os
import cv2
from tqdm import tqdm

# Configruations
data_dirs = ['/media/heraqi/data/heraqi/int-end-to-end-ad/auc.carla.dataset_01/train',
             '/media/heraqi/data/heraqi/int-end-to-end-ad/auc.carla.dataset_01/val']
out_width = 200
out_height = 88

# Main function
print("Reading images dirs ...")
cam1_image_files = []
cam2_image_files = []
cam3_image_files = []
for data_dir in data_dirs:
    episodes = [os.path.join(data_dir, o) for o in os.listdir(data_dir) if
                        os.path.isdir(os.path.join(data_dir, o))]
    for current_episode_dir in episodes:
        cam1_image_files.extend([current_episode_dir+'/'+f for f in os.listdir(current_episode_dir)
            if (f.startswith('CentralRGB_') and not f.endswith('_resized.png'))])
        cam2_image_files.extend([current_episode_dir + '/' + f for f in os.listdir(current_episode_dir)
            if (f.startswith('RightRGB_') and not f.endswith('_resized.png'))])
        cam3_image_files.extend([current_episode_dir + '/' + f for f in os.listdir(current_episode_dir)
            if (f.startswith('LeftRGB_') and not f.endswith('_resized.png'))])

print("Resizing cam 1/3 images ...")
for img_file in tqdm(cam1_image_files):
    img = cv2.imread(img_file)
    if img is None:
        print(img_file)
        continue
    img_resized = cv2.resize(img, (out_width, out_height))
    scipy.misc.imsave(img_file.replace('.png', '_resized.png'), cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))

print("Resizing cam 2/3 images ...")
for img_file in tqdm(cam2_image_files):
    img = cv2.imread(img_file)
    if img is None:
        print(img_file)
        continue
    img_resized = cv2.resize(img, (out_width, out_height))
    scipy.misc.imsave(img_file.replace('.png', '_resized.png'), cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))

print("Resizing cam 3/3 images ...")
for img_file in tqdm(cam3_image_files):
    img = cv2.imread(img_file)
    if img is None:
        print(img_file)
        continue
    img_resized = cv2.resize(img, (out_width, out_height))
    scipy.misc.imsave(img_file.replace('.png', '_resized.png'), cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))

# Hack
images_files = ['/media/heraqi/data/heraqi/int-end-to-end-ad/auc.carla.dataset_01/train/episode_00716/RightRGB_00496.png',
'/media/heraqi/data/heraqi/int-end-to-end-ad/auc.carla.dataset_01/train/episode_00948/RightRGB_00535.png',
'/media/heraqi/data/heraqi/int-end-to-end-ad/auc.carla.dataset_01/train/episode_00189/RightRGB_01003.png',
'/media/heraqi/data/heraqi/int-end-to-end-ad/auc.carla.dataset_01/train/episode_00777/RightRGB_00388.png',
'/media/heraqi/data/heraqi/int-end-to-end-ad/auc.carla.dataset_01/train/episode_00906/RightRGB_00338.png',
'/media/heraqi/data/heraqi/int-end-to-end-ad/auc.carla.dataset_01/val/episode_00043/RightRGB_00164.png',
'/media/heraqi/data/heraqi/int-end-to-end-ad/auc.carla.dataset_01/train/episode_00245/LeftRGB_00938.png',
'/media/heraqi/data/heraqi/int-end-to-end-ad/auc.carla.dataset_01/train/episode_00452/LeftRGB_00091.png',
'/media/heraqi/data/heraqi/int-end-to-end-ad/auc.carla.dataset_01/train/episode_00637/LeftRGB_00294.png',
'/media/heraqi/data/heraqi/int-end-to-end-ad/auc.carla.dataset_01/val/episode_00047/LeftRGB_00275.png']
for img_file in images_files:
    dir = os.path.dirname(img_file)
    img = os.path.basename(img_file)
    number = int(re.search(r'\d+', img).group(0))
    number = str(number).zfill(5)
    
    files = [img_file,
             dir + "/CentralDepth_" + number + ".png",
             dir + "/CentralRGB_" + number + ".png",
             dir + "/CentralSemanticSeg_" + number + ".png",
             dir + "/LeftDepth_" + number + ".png",
             dir + "/LeftRGB_" + number + ".png",
             dir + "/LeftSemanticSeg_" + number + ".png",
             dir + "/RightDepth_" + number + ".png",
             dir + "/RightRGB_" + number + ".png",
             dir + "/RightSemanticSeg_" + number + ".png",
             dir + "/Lidar32_" + number + ".png",
             dir + "/Lidar32_" + number + ".ply",
             dir + "/measurements_" + number + ".json",
            ]
    for f in files:
        os.remove(f) if os.path.exists(f) else print("Couldn't find file: " + f)

    