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


import cv2
import numpy as np
import os
import time

# =====================================================
# CALIBRATION (AUTO CAPTURE, IMAGES PROPRES)
# =====================================================
def calibrateCamera(num_images=20, save_dir="calibration_images", cooldown=1.0):

    os.makedirs(save_dir, exist_ok=True)

    patterns = [(9,7), (7,9), (7,5), (4,6), (6,4), (5,7)]

    cam = cv2.VideoCapture(1)

    if not cam.isOpened():
        print("Camera 1 not found, trying index 0...")
        cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Camera error")
        return

    print("Auto-capture active — Q pour quitter")

    count = 0
    last_capture_time = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame_raw = frame.copy()  # ✅ image brute pour sauvegarde

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)

        detected = False
        detected_pattern = None
        detected_corners = None

        for checkerboard in patterns:
            found, corners = cv2.findChessboardCornersSB(
                gray,
                checkerboard,
                flags=cv2.CALIB_CB_NORMALIZE_IMAGE |
                      cv2.CALIB_CB_EXHAUSTIVE |
                      cv2.CALIB_CB_ACCURACY
            )

            if found:
                detected = True
                detected_pattern = checkerboard
                detected_corners = corners
                break

        display = frame.copy()  # image pour affichage uniquement

        if detected:
            cv2.drawChessboardCorners(display, detected_pattern, detected_corners, True)
            cv2.putText(display,
                        f"Detected {detected_pattern} | {count}/{num_images}",
                        (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,255,0),
                        2)

            now = time.time()
            if count < num_images and (now - last_capture_time) > cooldown:
                filename = os.path.join(save_dir, f"img_{count:02d}.jpg")
                cv2.imwrite(filename, frame_raw)  # 💾 image propre
                print(f"[+] Capture propre {count+1}/{num_images} : {filename}")
                count += 1
                last_capture_time = now

        else:
            cv2.putText(display,
                        f"Not detected | {count}/{num_images}",
                        (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,0,255),
                        2)

        cv2.imshow("Detection + Auto Capture (images propres)", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if count >= num_images:
            print("Nombre d'images atteint ✔️")
            break

    cam.release()
    cv2.destroyAllWindows()

#calibrateCamera()

import cv2
import numpy as np
import glob
from scipy.optimize import least_squares

CHECKERBOARD = (9, 6)
square_size = 1.0  # unité arbitraire (cm par ex)

# 1. Points 3D du damier (plan Z=0)
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints = []

images = glob.glob("calibration_images/*.jpg")

# 2. Détection des coins
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    if ret:
        corners = cv2.cornerSubPix(
            gray, corners, (11,11), (-1,-1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        )
        objpoints.append(objp)
        imgpoints.append(corners.reshape(-1, 2))

# 3. Homographies
Hs = []
for i in range(len(objpoints)):
    H, _ = cv2.findHomography(objpoints[i][:, :2], imgpoints[i])
    Hs.append(H)

# 4. Construction de V pour estimer K (méthode de Zhang)
def v_ij(H, i, j):
    return np.array([
        H[0,i]*H[0,j],
        H[0,i]*H[1,j] + H[1,i]*H[0,j],
        H[1,i]*H[1,j],
        H[2,i]*H[0,j] + H[0,i]*H[2,j],
        H[2,i]*H[1,j] + H[1,i]*H[2,j],
        H[2,i]*H[2,j],
    ])

V = []
for H in Hs:
    V.append(v_ij(H, 0, 1))
    V.append(v_ij(H, 0, 0) - v_ij(H, 1, 1))

V = np.array(V)
_, _, VT = np.linalg.svd(V)
b = VT[-1]

# 5. Récupération de K
B11, B12, B22, B13, B23, B33 = b
v0 = (B12*B13 - B11*B23) / (B11*B22 - B12**2)
lam = B33 - (B13**2 + v0*(B12*B13 - B11*B23)) / B11
alpha = np.sqrt(lam / B11)
beta  = np.sqrt(lam * B11 / (B11*B22 - B12**2))
gamma = -B12 * alpha**2 * beta / lam
u0    = gamma * v0 / beta - B13 * alpha**2 / lam

K = np.array([
    [alpha, gamma, u0],
    [0,     beta,  v0],
    [0,     0,     1]
])

print("Matrice intrinsèque K :\n", K)