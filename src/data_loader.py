# veri yukleme ve hazirlik

import numpy as np
import cv2

def load_lidar_points(file_path): # lidar bilgilerini alma fonksiyonu
    points = np.fromfile(file_path, dtype=np.float32) # cok veri var, yavaslama olmamasi icin
    points = points.reshape(-1, 4) # diskten okunan veri tek boyutlu dizi => (N, 4) matrisine cevirme
    return points # 4 sutun = X (ileri yon), Y (sol yon), Z (yukari yon), R (lazer sinyal siddeti)

def load_image(file_path): # kamera goruntusunu okuma ve RGB'e donusturme
    img = cv2.imread(file_path)
    if img is None:
        raise FileNotFoundError(f"Gorsel bulunamadi")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

def load_calibration(file_path): # P2, R0_rect ve Tr_velo_to_cam matrislerini dondurme
    raw_data = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, val = line.split(':', 1)
            raw_data[key] = np.array([float(x) for x in val.split()])

    P2 = raw_data['P2'].reshape(3, 4) # P2 matrisi = kamera koordinatlarini piksel koordinatlara donusturur

    R0_rect = np.eye(4, dtype=np.float32) # 3x3'ten 4x4 homojen matrise cevir
    R0_rect[:3, :3] = raw_data['R0_rect'].reshape(3, 3) # rotasyon (aci farki) matrisi, stereo dogrultma

    Tr_velo_to_cam = np.eye(4, dtype=np.float32) # dissal matris = lidar koordinatlari kameranin optik merkezine tasinir
    Tr_velo_to_cam[:3, :4] = raw_data['Tr_velo_to_cam'].reshape(3, 4)

    return {
        'P2' : P2,
        'R0_rect' : R0_rect,
        'Tr_velo_to_cam' : Tr_velo_to_cam
    }

if __name__ == "__main__":
    # test 1: lidar
    test_bin_path = r"C:\Coding\cam_lidar_fusion\kitti_tiny\training\velodyne\000000.bin"
    pts = load_lidar_points(test_bin_path)
    
    print("LiDAR Veri Tipi   :", type(pts))
    print("Nokta Bulutu Şekli:", pts.shape)
    print("İlk Nokta [X,Y,Z,R]:", pts[0])

    # test 2: img
    test_img_path = r"C:\Coding\cam_lidar_fusion\kitti_tiny\training\image_2\000000.jpeg"
    img = load_image(test_img_path)

    print(f"Görsel Şekli : {img.shape} (Yükseklik, Genişlik, Kanal)")
    print(f"Piksel Tipi  : {img.dtype}")

    # test 3: calib
    test_calib_path = r"C:\Coding\cam_lidar_fusion\kitti_tiny\training\calib\000000.txt"
    calib = load_calibration(test_calib_path)

    print("P2 Boyutu          :", calib['P2'].shape)
    print("R0_rect Boyutu     :", calib['R0_rect'].shape)
    print("Tr_velo_to_cam Boyutu:", calib['Tr_velo_to_cam'].shape)