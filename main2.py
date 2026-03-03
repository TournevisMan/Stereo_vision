import cv2
import numpy as np
import os
import time
import glob
from scipy.optimize import least_squares

def calibrateStereo(num_images=20, save_dir="stereo_calib"):

    os.makedirs(save_dir + "/left", exist_ok=True)
    os.makedirs(save_dir + "/right", exist_ok=True)

    camL = cv2.VideoCapture(1)
    camR = cv2.VideoCapture(2)

    if not camL.isOpened() or not camR.isOpened():
        print("Deux caméras requises.")
        return

    print("ESPACE = capture paire | Q = quitter")

    count = 0

    while True:
        retL, frameL = camL.read()
        retR, frameR = camR.read()

        if not retL or not retR:
            break

        display = np.hstack([frameL, frameR])
        cv2.imshow("Stereo Capture", display)

        key = cv2.waitKey(1) & 0xFF

        if key == 32 and count < num_images:
            cv2.imwrite(f"{save_dir}/left/img_{count:02d}.jpg", frameL)
            cv2.imwrite(f"{save_dir}/right/img_{count:02d}.jpg", frameR)
            print(f"Paire {count+1}/{num_images} capturée")
            count += 1

        if key == ord('q') or count >= num_images:
            break

    camL.release()
    camR.release()
    cv2.destroyAllWindows()

def compute_reprojection_error(objpoints, imgpoints, K, dist=np.zeros(5)):

    total_error = 0
    total_points = 0

    # Calculer rvec et tvec pour chaque image
    rvecs = []
    tvecs = []

    for objp, imgp in zip(objpoints, imgpoints):
        # solvePnP pour récupérer rvec et tvec
        objp = objp.astype(np.float32)
        imgp = imgp.astype(np.float32)

        ret, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist)
        if not ret:
            continue

        rvecs.append(rvec)
        tvecs.append(tvec)

        # Reprojection des points
        imgpoints_proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        imgpoints_proj = imgpoints_proj.reshape(-1, 2)

        # Erreur moyenne pour cette image
        error = np.linalg.norm(imgp - imgpoints_proj, axis=1).mean()
        total_error += error * len(objp)
        total_points += len(objp)

    mean_error = total_error / total_points
    print(f"Erreur de reprojection moyenne : {mean_error:.4f} pixels")
    return mean_error, rvecs, tvecs

import cv2
import numpy as np
import glob

def calibrate_intrinsic_zhang(image_folder, CHECKERBOARD=(7,5), square_size=1.0):

    # =============================
    # 1. Points 3D du damier (Z=0)
    # =============================
    objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints = []

    images = glob.glob(image_folder + "/*.jpg")

    # =============================
    # 2. Détection des coins
    # =============================
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

    if len(objpoints) < 3:
        raise ValueError("Pas assez d'images valides pour calibration.")

    # =============================
    # 3. Homographies
    # =============================
    Hs = []
    for i in range(len(objpoints)):
        H, _ = cv2.findHomography(objpoints[i][:, :2], imgpoints[i])
        Hs.append(H)

    # =============================
    # 4. Construction matrice V
    # =============================
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

    # =============================
    # 5. Résolution SVD
    # =============================
    _, _, VT = np.linalg.svd(V)
    b = VT[-1]

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

    return K, objpoints, imgpoints

def stereoCalibration():

    # =============================
    # Calibration intrinsèque manuelle
    # =============================
    K1, objpointsL, imgpointsL = calibrate_intrinsic_zhang("cam2_images")
    K2, objpointsR, imgpointsR = calibrate_intrinsic_zhang("cam1_images")

    err1 = compute_reprojection_error(objpointsL, imgpointsL, K1)
    err2 = compute_reprojection_error(objpointsR, imgpointsR, K2)
    print(f"Erreur de reprojection caméra gauche : {err1[0]:.4f} pixels")
    print(f"Erreur de reprojection caméra droite : {err2[0]:.4f} pixels")

    # Taille image
    sample_img = cv2.imread("cam2_images/img_00.jpg")
    h, w = sample_img.shape[:2]

    # Distorsion supposée nulle (car non estimée ici)
    dist1 = np.zeros(5)
    dist2 = np.zeros(5)

    # =============================
    # Calibration extrinsèque stéréo
    # =============================
    ret, K1, dist1, K2, dist2, R, T, E, F = cv2.stereoCalibrate(
        objpointsL,
        [p.reshape(-1,1,2) for p in imgpointsL],
        [p.reshape(-1,1,2) for p in imgpointsR],
        K1, dist1,
        K2, dist2,
        (w,h),
        flags=cv2.CALIB_FIX_INTRINSIC
    )

    print("\nRotation R :\n", R)
    print("\nTranslation T :\n", T)
    print("\nMatrice Essentielle E :\n", E)
    print("\nMatrice Fondamentale F :\n", F)

    return K1, K2, R, T, E, F

stereoCalibration()