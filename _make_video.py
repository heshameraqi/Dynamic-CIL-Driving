import cv2
import os
from ctypes import wintypes, windll
from functools import cmp_to_key
from tqdm import tqdm

# ----------------------------------------------------------------------------------------------------------------------
image_folder = r'C:\Work\Software\CARLA\results\ogm'
video_name = '..\\dynamic_conditional_imitation_learning_deployed_to_test_town.avi'
FPS = 15
size_scale = 1.0

# ----------------------------------------------------------------------------------------------------------------------
def winsort(data):   # Sort files names list by name
    _StrCmpLogicalW = windll.Shlwapi.StrCmpLogicalW
    _StrCmpLogicalW.argtypes = [wintypes.LPWSTR, wintypes.LPWSTR]
    _StrCmpLogicalW.restype = wintypes.INT
    cmp_fnc = lambda psz1, psz2: _StrCmpLogicalW(psz1, psz2)
    return sorted(data, key=cmp_to_key(cmp_fnc))


images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images = winsort(images)
frame = cv2.imread(os.path.join(image_folder, images[1]))  # size as the second image, because in some of my generated images the first image is a special case
height, width, layers = frame.shape
width = int(size_scale*width)
height = int(size_scale*height)

fourcc = cv2.VideoWriter_fourcc(*'XVID')  # codec that works for you: XVID, MJPG, mp4v
video = cv2.VideoWriter(os.path.join(image_folder, video_name), fourcc, FPS, (width, height))

for image in tqdm(images):  # start from the second image, because in some of my generated images the first image is a special case
    img = cv2.resize(cv2.imread(os.path.join(image_folder, image)), (width, height))
    video.write(img)

video.release()
cv2.destroyAllWindows()

