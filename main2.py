import cv2
import numpy as np
import os
import time
import glob
from scipy.optimize import least_squares
import random

def calibrateStereo(num_images=20, save_dir=""):

    os.makedirs(save_dir + "/left_images", exist_ok=True)
    os.makedirs(save_dir + "/right_images", exist_ok=True)

    camL = cv2.VideoCapture(2)
    camR = cv2.VideoCapture(1)

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
            cv2.imwrite(f"{save_dir}/left_images/img_{count:02d}.jpg", frameL)
            cv2.imwrite(f"{save_dir}/right_images/img_{count:02d}.jpg", frameR)
            print(f"Paire {count+1}/{num_images} capturée")
            count += 1

        if key == ord('q') or count >= num_images:
            break

    camL.release()
    camR.release()
    cv2.destroyAllWindows()

import cv2
import numpy as np

def stereo_reprojection_error(objpoints, imgpointsL, imgpointsR, K1, K2, R, T, dist1=None, dist2=None):
    """
    Calcule l'erreur de reprojection moyenne pour une calibration stéréo.
    
    objpoints : liste de points 3D du monde (N,3) pour chaque image
    imgpointsL : liste de points 2D détectés dans la caméra gauche (N,2)
    imgpointsR : liste de points 2D détectés dans la caméra droite (N,2)
    K1, K2 : matrices intrinsèques des deux caméras
    R, T   : rotation et translation entre les caméras
    dist1, dist2 : vecteurs de distorsion (optionnels, défaut : zéro)
    
    Retourne :
        mean_error : erreur moyenne en pixels
        errors_L : erreurs par image pour la caméra gauche
        errors_R : erreurs par image pour la caméra droite
    """
    if dist1 is None:
        dist1 = np.zeros(5)
    if dist2 is None:
        dist2 = np.zeros(5)
        
    total_error = 0
    total_points = 0
    errors_L = []
    errors_R = []

    # Boucle sur toutes les images calibrées
    for objp, imgL, imgR in zip(objpoints, imgpointsL, imgpointsR):
        objp = objp.astype(np.float32)
        imgL = imgL.astype(np.float32)
        imgR = imgR.astype(np.float32)

        # --- Caméra gauche ---
        retL, rvecL, tvecL = cv2.solvePnP(objp, imgL, K1, dist1)
        imgL_proj, _ = cv2.projectPoints(objp, rvecL, tvecL, K1, dist1)
        imgL_proj = imgL_proj.reshape(-1,2)
        errorL = np.linalg.norm(imgL - imgL_proj, axis=1).mean()
        errors_L.append(errorL)

        # --- Caméra droite ---
        # Transformation vers la caméra droite
        objp_in_R = (R @ objp.T + T).T  # rotation + translation
        retR, rvecR, tvecR = cv2.solvePnP(objp_in_R, imgR, K2, dist2)
        imgR_proj, _ = cv2.projectPoints(objp_in_R, rvecR, tvecR, K2, dist2)
        imgR_proj = imgR_proj.reshape(-1,2)
        errorR = np.linalg.norm(imgR - imgR_proj, axis=1).mean()
        errors_R.append(errorR)

        total_error += errorL * len(objp) + errorR * len(objp)
        total_points += 2 * len(objp)

    mean_error = total_error / total_points
    print(f"Erreur de reprojection stéréo moyenne : {mean_error:.4f} pixels")
    return mean_error, errors_L, errors_R

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


def calibrate_intrinsic_zhang(image_folder, CHECKERBOARD=(5,7), square_size=3.0):

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
    K1, objpointsL, imgpointsL = calibrate_intrinsic_zhang("calibration_data/left_images")
    K2, objpointsR, imgpointsR = calibrate_intrinsic_zhang("calibration_data/right_images")

    print("droite : ")
    errsL = compute_reprojection_error(objpointsL, imgpointsL, K1)
    print("gauche : ")
    errsR = compute_reprojection_error(objpointsR, imgpointsR, K2)

    # Taille image
    sample_img = cv2.imread("calibration_data/left_images/img_12.jpg")
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

    mean_err, errsL, errsR = stereo_reprojection_error(objpointsL, imgpointsL, imgpointsR, K1, K2, R, T)

    print("\nRotation R :\n", R)
    print("\nTranslation T :\n", T)
    print("\nMatrice Essentielle E :\n", E)
    print("\nMatrice Fondamentale F :\n", F)

    return K1, K2, R, T, E, F, objpointsL, imgpointsL, objpointsR, imgpointsR

import numpy as np
import cv2

def compute_epilines(pts1, pts2, F):
    """
    Calcule les lignes épipolaires pour deux ensembles de points stéréo.
    
    pts1 : points dans l'image 1 (N,2)
    pts2 : points dans l'image 2 (N,2)
    F    : matrice fondamentale 3x3

    Retourne :
        lines1 : lignes dans l'image 1 correspondant aux pts2
        lines2 : lignes dans l'image 2 correspondant aux pts1
    """
    # Convertir en format (N,1,2) pour OpenCV
    pts1_cv = pts1.reshape(-1,1,2)
    pts2_cv = pts2.reshape(-1,1,2)

    # Lignes dans image 1 pour points de image 2
    lines1 = cv2.computeCorrespondEpilines(pts2_cv, 2, F).reshape(-1,3)
    # Lignes dans image 2 pour points de image 1
    lines2 = cv2.computeCorrespondEpilines(pts1_cv, 1, F).reshape(-1,3)

    return lines1, lines2

import random
import cv2

def draw_epilines(img1, img2, pts1, pts2, F):
    """
    Trace les lignes épipolaires pour deux images stéréo.

    img1 : image gauche
    img2 : image droite
    pts1 : points détectés dans img1
    pts2 : points détectés dans img2
    F    : matrice fondamentale 3x3

    Retourne les images avec lignes tracées.
    """

    def draw_lines(img, lines, pts):
        img_color = img.copy()
        for r_line, pt in zip(lines, pts):
            color = tuple([random.randint(0,255) for _ in range(3)])
            a, b, c = r_line
            # Calcul de deux points sur la ligne
            x0, y0 = 0, int(-c/b)
            x1, y1 = img.shape[1], int(-(c + a*img.shape[1])/b)
            cv2.line(img_color, (x0, y0), (x1, y1), color, 1)
            cv2.circle(img_color, tuple(pt.astype(int)), 5, color, -1)
        return img_color

    # Calcul des lignes épipolaires
    lines1, lines2 = compute_epilines(pts1, pts2, F)

    img1_lines = draw_lines(img1, lines1, pts1)
    img2_lines = draw_lines(img2, lines2, pts2)

    return img1_lines, img2_lines

import numpy as np

def epipolar_error(pts1, pts2, F):

    errors = []

    for p1, p2 in zip(pts1, pts2):

        x1 = np.array([p1[0], p1[1], 1])
        x2 = np.array([p2[0], p2[1], 1])

        error = abs(x2.T @ F @ x1)
        errors.append(error)

    return np.mean(errors)

def epipolar_distance(ptsL, ptsR, F):

    errors = []

    for pL, pR in zip(ptsL, ptsR):

        xL = np.array([pL[0], pL[1], 1])

        line = F @ xL

        a,b,c = line

        x,y = pR

        dist = abs(a*x + b*y + c) / np.sqrt(a*a + b*b)

        errors.append(dist)

    return np.mean(errors)

def show_chessboard_corners(image_folder, CHECKERBOARD=(7,5), delay=0):
    """
    Affiche les images d'un dossier avec les coins du damier détectés.

    image_folder : chemin vers le dossier contenant les images (*.jpg)
    CHECKERBOARD : tuple (colonnes, lignes) du damier
    delay        : temps d'attente entre images (0 = attente touche)
    """

    # Récupérer toutes les images JPG du dossier
    images = sorted(glob.glob(image_folder + "/*.jpg"))
    if not images:
        print("Aucune image trouvée dans le dossier.")
        return

    print(f"{len(images)} images trouvées, affichage des coins du damier...")

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Détection des coins
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        display_img = img.copy()

        if ret:
            # Affiner la détection
            corners_subpix = cv2.cornerSubPix(
                gray, corners, (11,11), (-1,-1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
            )
            # Dessiner les coins détectés
            cv2.drawChessboardCorners(display_img, CHECKERBOARD, corners_subpix, ret)
            status = "Coins détectés"
        else:
            status = "Aucun coin trouvé"

        # Afficher le nom de l'image et le statut
        cv2.putText(display_img, f"{fname.split('/')[-1]} - {status}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # Affichage
        cv2.imshow("Chessboard Corners", display_img)
        key = cv2.waitKey(delay) & 0xFF
        # Appuyer sur 'q' pour quitter
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    # Afficher toutes les images du dossier "calibration_data/left_images"
show_chessboard_corners("calibration_data/left_images", CHECKERBOARD=(7,5), delay=0)

#calibrateStereo(num_images=20, save_dir="calibration_data")
# Charger une paire d'images (ici la première paire)
imgL = cv2.imread("calibration_data/left_images/img_12.jpg")
imgR = cv2.imread("calibration_data/right_images/img_12.jpg")

# Récupérer les points détectés dans la première image de chaque caméra
K1, K2, R, T, E, F, objpointsL, imgpointsL, objpointsR, imgpointsR = stereoCalibration()

# Pour simplifier, on peut recalculer les coins du damier pour la première image
# ou utiliser ceux déjà détectés dans imgpointsL et imgpointsR
# Ici on suppose imgpointsL[0] et imgpointsR[0] existent après calibration


print("Direction T :", T / np.linalg.norm(T))

ptsL = imgpointsL[9]
ptsR = imgpointsR[9]

linesL, linesR = compute_epilines(ptsL, ptsR, F)

error = epipolar_distance(ptsL, ptsR, F)

print("Erreur épipolaire moyenne :", error)

# Tracer les lignes épipolaires
imgL_epi, imgR_epi = draw_epilines(imgL, imgR, ptsL, ptsR, F)

# Afficher les résultats
cv2.imshow("Left Image with Epilines", imgL_epi)
cv2.imshow("Right Image with Epilines", imgR_epi)
cv2.waitKey(0)
cv2.destroyAllWindows()