import cv2
import numpy as np

def cameraRecord(n): # n is the number of cameras
    if n < 1:
        print("Number of cameras must be at least 1.")
        return
    
    elif n == 1:

        cam = cv2.VideoCapture(1)
        frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))
        while True:
            ret, frame = cam.read()
            out.write(frame)
            cv2.imshow('Camera', frame)

            if cv2.waitKey(1) == ord('q'):
                break
        cam.release()
        out.release()
        cv2.destroyAllWindows()

    elif n == 2:
        # Open the default camera
        cam = cv2.VideoCapture(1)
        cam2 = cv2.VideoCapture(2)

        # Get the default frame width and height
        frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

        while True:
            ret, frame = cam.read(color=cv2.COLOR_BGR2GRAY)
            ret2, frame2 = cam2.read(color=cv2.COLOR_BGR2GRAY)

            # Write the frame to the output file
            out.write(frame)
            out.write(frame2)
            final = cv2.hconcat([frame, frame2])

            # Display the captured frame
            cv2.imshow('Camera', final)

            # Press 'q' to exit the loop
            if cv2.waitKey(1) == ord('q'):
                break

        # Release the capture and writer objects
        cam.release()
        cam2.release()
        out.release()
        cv2.destroyAllWindows()
"""
def calibrateCamera(img, R, T):
    # Define the camera matrix (intrinsic parameters)
    P = 

    # Project the 3D points to 2D image coordinates
    

    return K

"""
cameraRecord(2)
