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

def calibrateCamera(num_images=20):

    # Taille de la mire (nombre de coins internes)
    checkerboard = (9, 6)

    # Critère d'arrêt pour affiner les coins
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Points 3D réels dans le monde
    objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard[0],
                           0:checkerboard[1]].T.reshape(-1, 2)

    objpoints = []  # points 3D
    imgpoints = []  # points 2D

    cam = cv2.VideoCapture(1)

    captured = 0

    print("Montre la mire à la caméra.")
    print("Appuie sur 'c' pour capturer une image.")
    print("Appuie sur 'q' pour quitter.")

    while captured < num_images:

        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Détection des coins
        ret_cb, corners = cv2.findChessboardCorners(gray, checkerboard, None)

        if ret_cb:
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )
            cv2.drawChessboardCorners(frame, checkerboard, corners2, ret_cb)

        cv2.imshow("Calibration", frame)

        key = cv2.waitKey(1)

        # Capture avec touche 'c'
        if key == ord('c') and ret_cb:
            objpoints.append(objp)
            imgpoints.append(corners2)
            captured += 1
            print(f"Image capturée : {captured}/{num_images}")

        if key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    if len(objpoints) < 5:
        print("Pas assez d'images pour calibrer.")
        return

    # Calibration
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    print("\n=== Résultats calibration ===")
    print("Matrice caméra (K):\n", K)
    print("\nCoefficients de distorsion:\n", dist)
    print("\nErreur RMS:", ret)

    # Sauvegarde
    np.savez("camera_calibration.npz", K=K, dist=dist)

    print("\nCalibration sauvegardée dans camera_calibration.npz")

# Lance la calibration
calibrateCamera()
