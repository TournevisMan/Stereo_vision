import cv2
import numpy as np

def cameraRecord(n):  # n is the number of cameras
    if n < 1:
        print("Number of cameras must be at least 1.")
        return

    elif n == 1:

        cam = cv2.VideoCapture(1)

        if not cam.isOpened():
            print("Camera 1 not found, trying index 0...")
            cam = cv2.VideoCapture(0)

        if not cam.isOpened():
            print("No camera detected.")
            return

        frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

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

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            out.write(gray)
            cv2.imshow('Camera (Gray)', gray)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cam.release()
        out.release()
        cv2.destroyAllWindows()

    elif n == 2:
        cam = cv2.VideoCapture(1)
        cam2 = cv2.VideoCapture(2)

        if not cam.isOpened():
            cam = cv2.VideoCapture(0)

        if not cam2.isOpened():
            cam2 = cv2.VideoCapture(1)

        if not cam.isOpened() or not cam2.isOpened():
            print("Two cameras not detected.")
            return

        frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

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

            gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            final = cv2.hconcat([gray1, gray2])

            out.write(final)
            cv2.imshow('Camera (Gray)', final)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cam.release()
        cam2.release()
        out.release()
        cv2.destroyAllWindows()


# =====================================================
# CALIBRATION (même structure mais améliorée)
# =====================================================
def calibrateCamera(num_images=20):

    patterns = [(9,7), (7,9), (7,5), (4,6), (6,4), (5,7)]

    cam = cv2.VideoCapture(1)

    if not cam.isOpened():
        print("Camera 1 not found, trying index 0...")
        cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Camera error")
        return

    print("Press Q to quit")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 🔥 amélioration légère et utile (meilleur que equalizeHist)
        gray = cv2.GaussianBlur(gray, (5,5), 0)

        detected = False

        for checkerboard in patterns:

            found, corners = cv2.findChessboardCornersSB(
                gray,
                checkerboard,
                flags=cv2.CALIB_CB_NORMALIZE_IMAGE |
                      cv2.CALIB_CB_EXHAUSTIVE |
                      cv2.CALIB_CB_ACCURACY
            )

            if found:
                cv2.drawChessboardCorners(frame, checkerboard, corners, found)

                cv2.putText(frame,
                            f"Detected: {checkerboard}",
                            (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0,255,0),
                            2)

                detected = True
                break

        if not detected:
            cv2.putText(frame,
                        "Not detected",
                        (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,0,255),
                        2)

        cv2.imshow("Detection Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


# Lance la calibration
calibrateCamera()