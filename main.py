import cv2
import numpy as np

def cameraRecord(n):  # n is the number of cameras
    if n < 1:
        print("Number of cameras must be at least 1.")
        return

    elif n == 1:

        cam = cv2.VideoCapture(1)

        frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # isColor=False pour grayscale
        out = cv2.VideoWriter(
            'output.mp4',
            fourcc,
            20.0,
            (frame_width, frame_height),
            isColor=False
        )

        while True:
            ret, frame = cam.read()
            if not ret:
                break

            # Conversion en grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            out.write(gray)
            cv2.imshow('Camera (Gray)', gray)

            if cv2.waitKey(1) == ord('q'):
                break

        cam.release()
        out.release()
        cv2.destroyAllWindows()

    elif n == 2:
        cam = cv2.VideoCapture(1)
        cam2 = cv2.VideoCapture(2)

        frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Largeur doublée car concaténation horizontale
        out = cv2.VideoWriter(
            'output.mp4',
            fourcc,
            20.0,
            (frame_width * 2, frame_height),
            isColor=False
        )

        while True:
            ret, frame = cam.read()
            ret2, frame2 = cam2.read()

            if not ret or not ret2:
                break

            # Conversion en grayscale
            gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            final = cv2.hconcat([gray1, gray2])

            out.write(final)
            cv2.imshow('Camera (Gray)', final)

            if cv2.waitKey(1) == ord('q'):
                break

        cam.release()
        cam2.release()
        out.release()
        cv2.destroyAllWindows()

cameraRecord(1)
