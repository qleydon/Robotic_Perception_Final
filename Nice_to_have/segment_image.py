from gluoncv import model_zoo
import cv2
from gluoncv import utils
import numpy as np
from mxnet import nd
from mxnet.image import imread
from mxnet.image import imresize
# SiamRPN model
siamrpn = model_zoo.get_model('siamrpn_alexnet_v2_otb15', pretrained=True)

# Mask R-CNN model
maskrcnn = model_zoo.get_model('mask_rcnn_resnet50_v1b_coco', pretrained=True)

# get bounding box
cap = cv2.VideoCapture('test_video.avi')
ret, frame = cap.read()
frame_height, frame_width = frame.shape[:2]
print(frame_height, frame_width)

if not ret:
    print('cannot read the video')

# Select the bounding box in the first frame
bbox = cv2.selectROI(frame, False)

# Initialize SiamRPN with the bounding box
x, y, w, h = map(int, bbox)
tracked_person = frame[y:y + h, x:x + w]

tracked_person_nd = imresize(nd.array(tracked_person), w=11, h=3).expand_dims(axis=0)# Print the shape of the template_frame
print("Template Frame Shape:", tracked_person_nd.shape)

# Use the template method with the template image as a keyword argument
siamrpn.template(zinput=tracked_person_nd)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # SiamRPN tracking
    bbox = siamrpn.track(frame)

    # Extract the object using the bounding box
    x, y, w, h = map(int, bbox)
    tracked_object = frame[y:y + h, x:x + w]

    # Apply Mask R-CNN for segmentation
    mask, _ = maskrcnn(utils.transform(tracked_object))

    # Visualize or use the tracking and segmentation results as needed
    # ...

    # Display the frame (you might want to modify this based on your needs)
    cv2.imshow('Tracking and Segmentation', frame)
    if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()


