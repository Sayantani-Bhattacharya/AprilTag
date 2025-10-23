## To activate python environment:
pyenv activate realsense-env

Using python downgraded version of Python 3.10.13 from Python 3.12.3

## static_tag_detection: 
- RS as input frame, pose tracking in cv show,family: tag16h5 </br>
- using downgraded python virtual env. </br>
- Need to log the path-pose trajectory in csv.

## dyn_tag_detection:
- The downloaded video from blue OS Cockpit interface as input frame. pose tracking in cv show,family: tag16h5 </br>
- using downgraded python virtual env. </br>
- log the path-pose trajectory in csv for all visible tags.
- [for 18 dynamic and 2 static markers: in case of mono rov cam]

## 6dPoseVisulalize.py:
- Reads and plots the 3d pose path trajectory of the detected tags.

## extDynDet
- uses the direct real sense topics and does real-time pose detection. 
- Need to save array, and transforms of relative pose, and plot stuff.

## Works but Not used.
## apriltag_video.py:
- input is mp4, downloaded and modified from BlueOS Cockpit. </br>
- Need to modify to make it work for tagStandard41h12 </br>
- Need to log the path-pose trajectory in csv for 18 dynamic and 2 static markers.

## apriltag_image.py:
- input is jpg, downloaded and modified from BlueOS Cockpit. </br>
- Need to modify to make it work for tagStandard41h12 </br>
- Mainly for debugging filters in a single frame.
