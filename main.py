import cv2
import numpy as np
import os
import pickle


# =====================================================
# 1️⃣ CAPTURE DES IMAGES
# =====================================================
def captureCalibrationImages(num_images=20, save_dir="calibration_images"):

    os.makedirs(save_dir, exist_ok=True)

    checkerboard = (7,5)  # ⚠️ adapte à ton damier

    cam = cv2.VideoCapture(1)

    if not cam.isOpened():
        cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Camera error")
        return

    print("ESPACE = capturer | Q = quitter")

    count = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCornersSB(gray, checkerboard)

        if found:
            #cv2.drawChessboardCorners(frame, checkerboard, corners, True)
            cv2.putText(frame, f"Detected | {count}/{num_images}",
                        (20,40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,255,0), 2)
        else:
            cv2.putText(frame, f"Not detected | {count}/{num_images}",
                        (20,40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,0,255), 2)

        cv2.imshow("Capture Calibration", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 32 and found and count < num_images:
            filename = os.path.join(save_dir, f"img_{count:02d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"[+] Capture {count+1}/{num_images}")
            count += 1

        if key == ord('q') or count >= num_images:
            break

    cam.release()
    cv2.destroyAllWindows()


# =====================================================
# 2️⃣ ANALYSE DES IMAGES (DETECTION + SAUVEGARDE POINTS)
# =====================================================
def analyzeCalibrationImages(save_dir="calibration_images",
                             checkerboard=(7,5),
                             output_file="calibration_data.pkl"):

    objpoints = []
    imgpoints = []

    objp = np.zeros((checkerboard[0]*checkerboard[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:checkerboard[0],
                          0:checkerboard[1]].T.reshape(-1,2)

    images = sorted(os.listdir(save_dir))

    if len(images) == 0:
        print("Aucune image trouvée.")
        return

    valid_images = 0

    for fname in images:

        img_path = os.path.join(save_dir, fname)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCornersSB(gray, checkerboard)

        if found:
            objpoints.append(objp)
            imgpoints.append(corners)
            valid_images += 1
        else:
            print(f"Damier non détecté dans {fname}")

    print(f"{valid_images} images valides trouvées.")

    # 🔥 Sauvegarde pour réutilisation future
    with open(output_file, "wb") as f:
        pickle.dump({
            "objpoints": objpoints,
            "imgpoints": imgpoints,
            "image_size": gray.shape[::-1]
        }, f)

    print(f"Données sauvegardées dans {output_file}")


# =====================================================
# 3️⃣ CALIBRATION À PARTIR DES DONNÉES SAUVEGARDÉES
# =====================================================
def runCalibration(data_file="calibration_data.pkl"):

    if not os.path.exists(data_file):
        print("Fichier de données introuvable.")
        return

    with open(data_file, "rb") as f:
        data = pickle.load(f)

    objpoints = data["objpoints"]
    imgpoints = data["imgpoints"]
    image_size = data["image_size"]

    if len(objpoints) < 5:
        print("Pas assez d'images valides pour calibrer.")
        return

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None
    )

    print("\n========== RESULTATS ==========")
    print("Erreur RMS :", ret)
    print("\nMatrice caméra :\n", mtx)
    print("\nCoefficients de distorsion :\n", dist)

    np.savez("camera_calibration.npz",
             camera_matrix=mtx,
             distortion=dist)

    print("Paramètres sauvegardés dans camera_calibration.npz ✔️")

    return mtx, dist


# =====================================================
# 📷 CAPTURE STÉRÉO (2 CAMÉRAS EN MÊME TEMPS)
# =====================================================
def captureStereoCalibrationImages(num_images=20,
                                   left_dir="left_images",
                                   right_dir="right_images"):

    os.makedirs(left_dir, exist_ok=True)
    os.makedirs(right_dir, exist_ok=True)

    checkerboard = (7,5)

    camR = cv2.VideoCapture(1)
    camL = cv2.VideoCapture(2)

    if not camL.isOpened() or not camR.isOpened():
        print("Erreur ouverture caméras")
        return

    print("ESPACE = capturer | Q = quitter")

    count = 0

    while True:

        retL, frameL = camL.read()
        retR, frameR = camR.read()

        if not retL or not retR:
            print("Erreur lecture caméra")
            break

        grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)

        foundL, cornersL = cv2.findChessboardCornersSB(grayL, checkerboard)
        foundR, cornersR = cv2.findChessboardCornersSB(grayR, checkerboard)

        # Affichage statut
        #status = "Detected" if foundL and foundR else "Not detected"

        color = (0,255,0) if foundL and foundR else (0,0,255)

        #cv2.putText(frameL, f"{status} | {count}/{num_images}",
                    #(20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        #cv2.putText(frameR, f"{status} | {count}/{num_images}",
                    #(20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Affichage côte à côte
        combined = np.hstack((frameL, frameR))
        cv2.imshow("Stereo Capture", combined)

        key = cv2.waitKey(1) & 0xFF

        if key == 32 and foundL and foundR and count < num_images:

            filenameL = os.path.join(left_dir, f"img_{count:02d}.jpg")
            filenameR = os.path.join(right_dir, f"img_{count:02d}.jpg")

            cv2.imwrite(filenameL, frameL)
            cv2.imwrite(filenameR, frameR)

            print(f"[+] Capture paire {count+1}/{num_images}")
            count += 1

        if key == ord('q') or count >= num_images:
            break

    camL.release()
    camR.release()
    cv2.destroyAllWindows()


def runStereoCalibration(left_dir="left_images",
                         right_dir="right_images",
                         checkerboard=(7,5),
                         square_size=0.03):  # 30 mm = 0.03 m

    print("\n===== CALIBRATION STÉRÉO =====")

    # 🔥 Tri numérique sécurisé
    imagesL = sorted(os.listdir(left_dir),
                     key=lambda x: int(x.split('_')[1].split('.')[0]))
    imagesR = sorted(os.listdir(right_dir),
                     key=lambda x: int(x.split('_')[1].split('.')[0]))

    imagesL = [os.path.join(left_dir, f) for f in imagesL]
    imagesR = [os.path.join(right_dir, f) for f in imagesR]

    if len(imagesL) == 0 or len(imagesR) == 0:
        print("Images manquantes.")
        return

    if len(imagesL) != len(imagesR):
        print("Nombre d'images gauche/droite différent.")
        return

    # 🔥 Taille image prise depuis la première image
    first_img = cv2.imread(imagesL[0])
    image_size = (first_img.shape[1], first_img.shape[0])

    objpoints = []
    imgpointsL = []
    imgpointsR = []

    # Points 3D du damier
    objp = np.zeros((checkerboard[0]*checkerboard[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:checkerboard[0],
                          0:checkerboard[1]].T.reshape(-1,2)
    objp *= square_size

    valid_pairs = 0

    for imgL_path, imgR_path in zip(imagesL, imagesR):

        imgL = cv2.imread(imgL_path)
        imgR = cv2.imread(imgR_path)

        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        foundL, cornersL = cv2.findChessboardCornersSB(grayL, checkerboard)
        foundR, cornersR = cv2.findChessboardCornersSB(grayR, checkerboard)

        if foundL and foundR:

            # 🔥 SB est déjà subpixel précis → pas de cornerSubPix
            objpoints.append(objp.copy())
            imgpointsL.append(cornersL)
            imgpointsR.append(cornersR)

            valid_pairs += 1
        else:
            print(f"Damier non détecté pour {imgL_path}")

    if valid_pairs < 5:
        print("Pas assez de paires valides.")
        return

    print(f"{valid_pairs} paires valides détectées.")

    # -----------------------------
    # Calibration individuelle
    # -----------------------------
    print("Calibration caméra gauche...")
    retL, mtxL, distL, _, _ = cv2.calibrateCamera(
        objpoints, imgpointsL, image_size, None, None)

    print("Calibration caméra droite...")
    retR, mtxR, distR, _, _ = cv2.calibrateCamera(
        objpoints, imgpointsR, image_size, None, None)

    # -----------------------------
    # Calibration stéréo
    # -----------------------------
    print("Calibration stéréo...")

    flags = cv2.CALIB_FIX_INTRINSIC

    retStereo, mtxL, distL, mtxR, distR, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpointsL,
        imgpointsR,
        mtxL,
        distL,
        mtxR,
        distR,
        image_size,
        flags=flags
    )

    print("\n===== RÉSULTATS =====")
    print("RMS Gauche :", retL)
    print("RMS Droite :", retR)

    print("\nRotation R :\n", R)
    print("\nTranslation T (mètres) :\n", T)

    # -----------------------------
    # Rectification
    # -----------------------------
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtxL, distL,
        mtxR, distR,
        image_size,
        R, T
    )

    print("Rectification calculée ✔")

    np.savez("stereo_calibration.npz",
             mtxL=mtxL,
             distL=distL,
             mtxR=mtxR,
             distR=distR,
             R=R,
             T=T,
             E=E,
             F=F,
             R1=R1,
             R2=R2,
             P1=P1,
             P2=P2,
             Q=Q)

    print("Paramètres sauvegardés dans stereo_calibration.npz ✔️")

    return mtxL, distL, mtxR, distR, R, T


def showEpipolarLines(left_dir="left_images",
                      right_dir="right_images",
                      calibration_file="stereo_calibration.npz"):

    print("\n===== AFFICHAGE LIGNES ÉPIPOLAIRES =====")

    # Charger paramètres
    data = np.load(calibration_file)

    mtxL = data["mtxL"]
    distL = data["distL"]
    mtxR = data["mtxR"]
    distR = data["distR"]
    R = data["R"]
    T = data["T"]
    R1 = data["R1"]
    R2 = data["R2"]
    P1 = data["P1"]
    P2 = data["P2"]

    # Charger première paire
    imagesL = sorted(os.listdir(left_dir),
                     key=lambda x: int(x.split('_')[1].split('.')[0]))
    imagesR = sorted(os.listdir(right_dir),
                     key=lambda x: int(x.split('_')[1].split('.')[0]))

    imgL = cv2.imread(os.path.join(left_dir, imagesL[0]))
    imgR = cv2.imread(os.path.join(right_dir, imagesR[0]))

    h, w = imgL.shape[:2]

    # Rectification maps
    mapLx, mapLy = cv2.initUndistortRectifyMap(
        mtxL, distL, R1, P1, (w, h), cv2.CV_32FC1)

    mapRx, mapRy = cv2.initUndistortRectifyMap(
        mtxR, distR, R2, P2, (w, h), cv2.CV_32FC1)

    rectL = cv2.remap(imgL, mapLx, mapLy, cv2.INTER_LINEAR)
    rectR = cv2.remap(imgR, mapRx, mapRy, cv2.INTER_LINEAR)

    # Fusion côte à côte
    combined = np.hstack((rectL, rectR))

    # Dessin lignes horizontales
    for y in range(0, h, 40):
        cv2.line(combined, (0, y), (2*w, y), (0,255,0), 1)

    cv2.imshow("Epipolar Lines", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#captureCalibrationImages()
captureStereoCalibrationImages()
# Analyse
#analyzeCalibrationImages()


# Calibration
#runCalibration()
runStereoCalibration()

#epipolar lines
#showEpipolarLines()