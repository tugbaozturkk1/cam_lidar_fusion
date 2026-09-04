# koordinat donusumu ve izdusum
# lidar noktalarinin (X, Y, Z), kameranin piksel koordinatlarina (u, v) ve onumuzdeki derinlige (z) donusumu

import sys
import os
import numpy as np
from data_loader import load_lidar_points, load_calibration, load_image

def prepare_lidar_points(pts_3d_velo): # ham lidar verisinden R'i cikarma (koordinat degil)
    points_xyz = pts_3d_velo[:, :3]
    ones = np.ones((points_xyz.shape[0], 1), dtype=np.float32)
    points_hom = np.hstack((points_xyz, ones))
    return points_hom

def compute_projection_matrix(calib):
    P2 = calib['P2']
    R0_rect = calib['R0_rect']
    Tr_velo_to_cam = calib['Tr_velo_to_cam']

    proj_mat = P2 @ R0_rect @ Tr_velo_to_cam # birlesik donusum matrisi => bir noktayi lidar'dan alip kamera piksel duzlemine firlatma

    return proj_mat

def project_to_camera_frame(points_hom, proj_mat):
    points_2d_hom = proj_mat @ points_hom.T
    # 0. satir: olceklenmis yatay piksel (u * z)
    # 1. satir: olceklenmis dikey piksel (v * z)
    # 2. satir: kameraya olan gercek derinlik (z - metre cinsinden mesafe)
    return points_2d_hom

def filter_points_in_img(points_2d, depths, img_shape):
    # img_shape = (H, W, 3) veya (H, W)
    height, width = img_shape[:2]

    u = points_2d[:, 0]
    v = points_2d[:, 1]

    mask = (depths > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    valid_points_2d = points_2d[mask]
    valid_depths = depths[mask]

    return valid_points_2d, valid_depths, mask

def normalize_to_pixel_coords(points_2d_hom):
    depths = points_2d_hom[2, :] # = z

    # homojen koordinatlardan piksel koordinatlarina gecis
    # u => goruntudeki yatay piksel koordinati, v => goruntudeki dikey piksel koordinati
    u = points_2d_hom[0, :] / depths
    v = points_2d_hom[1, :] / depths

    points_2d = np.vstack((u, v)).T
    return points_2d, depths

if __name__ == "__main__":
    # test 1 => homojen koordinata gecis dogrulamasi
    bin_path = r"C:\Coding\cam_lidar_fusion\kitti_tiny\training\velodyne\000000.bin"
    raw_points = load_lidar_points(bin_path)
    # (N, 3) boyutundaki [X, Y, Z] matrisi ile (N, 1) boyutundaki '1'leri yan yana birlestirme
    # her satir [X, Y, Z, 1] haline gelir => boyut: (N, 4)
    hom_points = prepare_lidar_points(raw_points)

    print("Ham Nokta Şekli    :", raw_points.shape)
    print("Homojen Nokta Şekli:", hom_points.shape)
    print("İlk Nokta Değeri   :", hom_points[0])

    # test 2 => matris carpimi ve kamera koordinatina izdusum dogrulamasi
    raw_points = load_lidar_points(bin_path)
    calib_path = r"C:\Coding\cam_lidar_fusion\kitti_tiny\training\calib\000000.txt"
    calib = load_calibration(calib_path)

    hom_points = prepare_lidar_points(raw_points)
    proj_mat = compute_projection_matrix(calib)
    points_cam = project_to_camera_frame(hom_points, proj_mat)

    print("Birlesik Projeksiyon Matrisi :", proj_mat.shape)
    print("Kamera Koordinatlarindaki Noktalar:", points_cam.shape)
    print("Ilk Noktanin [x, y, z] Degeri :", points_cam[:, 0])

    # test 3 => piksel normalizasyonu dogrulama
    pts_2d, depths = normalize_to_pixel_coords(points_cam)

    print("Piksel Koordinatlari Sekli :", pts_2d.shape)
    print("Derinlik Dizisi Sekli      :", depths.shape)
    print("Ilk Noktanin [u, v] Pikseli:", pts_2d[0])
    print("Ilk Noktanin Derinligi (m) :", depths[0])

    # test 4 => filtreleme ve maskeleme
    img_path = r"C:\Coding\cam_lidar_fusion\kitti_tiny\training\image_2\000000.jpeg"
    img = load_image(img_path)

    valid_pts_2d, valid_depths, mask = filter_points_in_img(pts_2d, depths, img.shape)
    print("Toplam LiDAR Noktasi       :", len(pts_2d))
    print("Görsele Düşen Nokta Sayisi :", len(valid_pts_2d))
    print("Görsele Düşen Ilk Nokta [u, v]:", valid_pts_2d[0])
    print("Görsele Düşen Ilk Derinlik (m):", valid_depths[0])