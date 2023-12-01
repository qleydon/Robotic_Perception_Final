import cv2
import sys
sys.path.insert(0, '.')
from PyMDNet.Robotic_Perception_Final.FastSAM.fastSAM_inference import FastSAM_segmentation
from PyMDNet.Robotic_Perception_Final.FastSAM.fastsam.model import FastSAM
from PyMDNet.Robotic_Perception_Final.FastSAM.fastsam.prompt import FastSAMPrompt



# Get the video file and read it
video = cv2.VideoCapture("/home/quinn/Robotics_Final/OpenCV/test_video.avi")
ret, frame = video.read()
if not ret:
    print('cannot read the video')
# Select the bounding box in the first frame
bbox = cv2.selectROI(frame, False)
bbox = [388, 289, 65, 236]
frame_height, frame_width = frame.shape[:2]
print(frame_height, frame_width)
frame1 = frame

SAM_model = FastSAM("/home/quinn/Robotics_Final/PyMDNet/Robotic_Perception_Final/FastSAM/weights/FastSAM-x.pt")
result = FastSAM_segmentation(frame1, bbox)
print(result.shape)

