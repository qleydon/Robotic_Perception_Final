import cv2

def save_first_frame(video_path, output_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file opened successfully
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    # Read the first frame
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print("Error: Unable to read the first frame.")
        cap.release()
        return

    # Save the first frame as a JPEG image
    cv2.imwrite(output_path, frame)

    # Release the video capture object
    cap.release()

    print(f"The first frame has been saved as {output_path}.")

# Specify the path to the video file
video_path = '/home/quinn/Robotics_Final/OpenCV/test_video.avi'

# Specify the output path for the JPEG image
output_path = '/home/quinn/Robotics_Final/OpenCV/test_video.jpg'

# Call the function to save the first frame
save_first_frame(video_path, output_path)
