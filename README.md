# Dynamic Conditional Imitation Learning for Autonomous Driving

> **Cite our Work:**
> 
> Hesham M. Eraqi, Mohamed N. Moustafa, Jens Honer. Dynamic Conditional Imitation Learning for Autonomous Driving. IEEE Transactions on Intelligent Transportation Systems, DOI: 10.1109/TITS.2022.3214079, November 2022. https://ieeexplore.ieee.org/document/9928072

- Paper Link: https://ieeexplore.ieee.org/document/9928072/

- Preprint paper: https://arxiv.org/abs/2211.11579

![Added Work Zones on CARLA](imgs/Work%20Zones.png?raw=true)

## Contents:
- [Demo Video](#Demo-Video)
- [Summary](#Summary)
- [To use the code](#To-Use-The-Code)
- [Backups](#Backups)
- [Environment Setup](#Environment-Setup)
- [Notes](#Notes)

## Demo Video:
A real-time video showing an example scenario visualizing our system driving a car in the test town. The ego-car detects two road blockages and dynamically estimate and follows new routes to eventually reach the designation successfully, better watched in Full-HD:

[![Video showing D-CIL method in action in test town](imgs/Results%20Video%20Thumbnail.png?raw=true)](https://www.youtube.com/watch?v=v3DaKJL-HCQ)

## Summary:
An extension to the Conditional Imitation Learning approach for Autonomous Driving is presented to tackle the challenges of lack of generalization, inconsistency against varying weather conditions, and inability to avoid unexpected static road blockages. The laser scanner input is fused with the regular camera streams, at the features level of the proposed Deep Learning model, to overcome the generalization and consistency challenges. A new efficient Occupancy Grid Mapping method is introduced, with improved runtime performance, memory utilization, and map accuracy, along with new algorithms for road blockages avoidance and global route planning to allow for dynamically detecting partial and full road blockages and guiding the vehicle to another route to reach the destination. Experimental results on CARLA simulator urban driving benchmark demonstrated the effectiveness of the proposed methods.

## To use the code:
1. Download and unzip the needed CARLA version(s) from the [Backups section](#Backups) below.
2. Set the path to CARLA simulator you want to train with or deploy to (or add the command to /home/your_user_name/.bashrc):
    ```
    export CARLA_PATH="/home/heraqi/CARLA_0.8.4/"
    ```
    These are the paths in my machines:
        Local Windows Machine:
        C:\Work\Software\CARLA\CARLA_0.8.2\
        C:\Work\Software\CARLA\CARLA_0.8.4_Working_Zones\WindowsNoEditor\
        University DGX Linux Server:
        /home/heraqi/CARLA_0.8.4/
        /home/heraqi/CARLA_0.8.4_Working_Zones/LinuxNoEditor/
    Note that if you are using PyCharm to run this project the export commands above won't help. Because when any process get created it inherit the environment variables from its parent process (the OS itself in your case). So add the environment variables manually in PyCharm from the Run\Debug Configuration window. Run > Edit Configurations > Environment Variables -> ... don't use double quotations for the values.
3. Run any script starting by underscore (_*.py), you will find a dedicated section in the file beginning for configuration parameters

## Backups:
- Models download link: https://drive.google.com/open?id=1wiyTzlVng6ONZmcTD_Czq4twMItUSBf2
- CARLA 0.8.4 Built with Working Zones (for Linux): https://drive.google.com/u/1/uc?id=1ypq0DqZDpvqw-NWzIJx545YUmvOYmUOS
  Copy PythonClient folder from: https://drive.google.com/u/1/uc?id=1uSQKWbR-KrczN6cbuPEXnnF6ZtVwzd8s (Similar to CARLA 0.8.2 one)
  For Windows: https://drive.google.com/a/aucegypt.edu/uc?id=1j9bmfX846-eyNBGu1qGuGIHjfZxP5J1D
- CARLA 0.8.4 original built:
  Linux: https://drive.google.com/a/aucegypt.edu/uc?id=128208paE094rZFl0Ro4vSJ61ZBaggDKV
  Windows: Build it from sources using the tutorial in 'Notes & Tutorial & Documentations.txt'
- Updated CARLA Maps with working zones (maps only to build CARLA package from Github): https://drive.google.com/u/1/uc?id=1Lbuy9yv0oySB7_HcGfUJe_DiVQx6x85E
  CARLA 0.8.4 source: https://github.com/heshameraqi/carla/tree/0.8.4
- Training dataset download link:
  - data not backuped yet
  - metadata pickles for fast loading:  https://drive.google.com/open?id=1q2uS6-P8iyydQ8ujRV9pOf0Ar-V4WGNn & https://drive.google.com/open?id=1K_Zg7_Kg6QZXBHR14BsAupp6vzBP2q4G

## Environment Setup:
- Install Anaconda (sh ./Anaconda3-2019.10-Linux-x86_64.sh)
- conda install -c conda-forge shapely
- conda install -c anaconda tensorflow-gpu
- conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0 -c pytorch
- conda install python=3.6
- pip install scipy=1.2

## Notes:
- Quick notes and documentation are in quick_notes.txt
- Download CARLA compiled versions from: https://github.com/carla-simulator/carla/releases. Then refer to their paths in the code configurations.
This project requires versions 0.9.X and 0.8.2/4
- Install pdflatex library before being able to run _draw_network*.py, to generate the network graph PDF file
- PyCharm SSH Notes:
    - In the run configuration options, add the environment variable DISPLAY=localhost:10.0
  (the right hand side should be obtained with the command echo $DISPLAY in the server side via SSH, if display doesn't 
  show up and pycharm gives error: Could not connect to any X display, this should be the problem to fix)
    - In parallel to PyCharm, open MobaXTerm and connect while X11 forwarding checkbox is enabled. Now PyCharm will forward
  the display to MobaXTerm X11 host, OR have Xming opened in parallel
- Useful Commands (simple but for quicker access):
    You can run any script starting by underscore in background via SSH:
    ```
    nohup python _*.py &> nohup1.out&
    ```
    
    Copy dataset to local Windows from remote Unix:
    ```
    del /Q /S C:\Work\Software\int-end-to-end-ad\data\carla\* && C:\Work\Programs\SSH\pscp -pw Hesham19 -r heraqi@10.67.84.8:/mnt/7A0C2F9B0C2F5185/heraqi/data/int-end-to-end-ad/auc.carla.dataset/* C:\Work\Software\int-end-to-end-ad\data\carla\
    ```
    
    Save video from images in Unix and copy video to Windows (make sure to set CARLA game fps):
    ```
    cd /mnt/7A0C2F9B0C2F5185/heraqi/data/int-end-to-end-ad/auc.carla.dataset/0/sensor.lidar.debug/; ffmpeg -framerate 15 -pattern_type glob -i "*.jpg" -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4
    C:\Work\Programs\SSH\pscp -pw Hesham19 -r heraqi@10.67.84.8:/mnt/7A0C2F9B0C2F5185/heraqi/data/int-end-to-end-ad/auc.carla.dataset/0/sensor.lidar.debug/output.mp4 C:\Work\Software\int-end-to-end-ad\data\carla\
    ```
    
    Open town map with OpenDrive ODR Viewer:
    ```
    /home/heraqi/scripts/int-end-to-end-ad/int-end-to-end-ad-carla/_tools/odrViewer64.1.9.1
    then map xodr file like this: /mnt/7A0C2F9B0C2F5185/heraqi/CARLA_0.9.4/CarlaUE4/Content/Carla/Maps/OpenDrive/Town01.xodr
    ```
    
    View map positions:
    ```
    python /mnt/7A0C2F9B0C2F5185/heraqi/CARLA_0.8.4/PythonClient/view_start_positions.py
    ```
    
    Useful client GUI keyboard shortcuts:
    ```Tab: change ego-car camera view```
    ```Backquote: change displayed sensor```
    ```R: Reset```
    ```C: Change conditions```
    ```G: Toggle GUI```
