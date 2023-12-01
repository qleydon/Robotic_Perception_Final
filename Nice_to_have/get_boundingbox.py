import cv2


# Get the video file and read it
video = cv2.VideoCapture("/home/quinn/Robotics_Final/OpenCV/test_video.avi")
ret, frame = video.read()
if not ret:
    print('cannot read the video')
# Select the bounding box in the first frame
bbox = cv2.selectROI(frame, False)
bbox = [388, 289, 65, 236]

print(bbox)