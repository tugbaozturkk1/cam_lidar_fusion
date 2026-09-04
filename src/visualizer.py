# derinlik haritasi olusturma
# kucuk z (yakindaki nesneler) => sicak renkler
# buyuk z (uzaktaki nesneler) => soguk renkler

import matplotlib.pyplot as plt
import numpy as np
import os
from data_loader import load_lidar_points, load_calibration, load_image
from projection import prepare_lidar_points, compute_projection_matrix, project_to_camera_frame, normalize_to_pixel_coords, filter_points_in_img

def plot_lidar_on_image(image, points_2d, depths, max_depth=80.0, save_path=None):
    # image => RGB kamera goruntusu
    # points_2d => goruntuye dusen piksel koordinatlari
    # max_depth (float) =>  renklendirmede kullanilacak tavan derinlik mesafesi

    fig, ax = plt.subplots(figsize=(15, 5)) # kitti goruntu boyutuna uygun buyukluk
    ax.imshow(image)

    scatter = ax.scatter(
        points_2d[:, 0],
        points_2d[:, 1],
        c = depths,
        cmap='jet_r', # renk paleti
        s=1.5,      # nokta boyutu
        alpha=0.7,  # nokta seffafligi
        vmin=0.0,   # min renk degeri, en sicak renk = kirmizi/bordo
        vmax=max_depth  # max renk degeri, en soguk renk bitisi
    )

    cbar = fig.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.05, shrink=0.6) # mesafe olcegi
    cbar.set_label('Mesafe (Metre)')

    ax.set_axis_off() # eksen cizgilerini kapatma
    ax.set_title("LiDAR - Kamera Sensor Fuzyonu (Derinlik Haritasi)")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Gorsel basariyla kaydedildi: {save_path}")

    plt.close(fig)

if __name__ == "__main__":
    pts_path = r"C:\Coding\cam_lidar_fusion\kitti_tiny\training\velodyne\000000.bin"
    img_path = r"C:\Coding\cam_lidar_fusion\kitti_tiny\training\image_2\000000.jpeg"
    calib_path = r"C:\Coding\cam_lidar_fusion\kitti_tiny\training\calib\000000.txt"

    pts = load_lidar_points(pts_path)
    img = load_image(img_path)
    calib = load_calibration(calib_path)

    # koordinat izdusumu ve filtreleme
    hom_pts = prepare_lidar_points(pts)
    proj_mat = compute_projection_matrix(calib)
    pts_cam = project_to_camera_frame(hom_pts, proj_mat)
    pts_2d, depths = normalize_to_pixel_coords(pts_cam)
    valid_pts_2d, valid_depths, _ = filter_points_in_img(pts_2d, depths, img.shape)

    output_path = r"C:\Coding\cam_lidar_fusion\results\fusion_000000.png"
    plot_lidar_on_image(img, valid_pts_2d, valid_depths, max_depth=80.0, save_path=output_path)