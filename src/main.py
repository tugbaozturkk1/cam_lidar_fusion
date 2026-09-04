import os
import glob
from data_loader import load_lidar_points, load_image, load_calibration
from projection import (
    prepare_lidar_points,
    compute_projection_matrix,
    project_to_camera_frame,
    normalize_to_pixel_coords,
    filter_points_in_img
)
from visualizer import plot_lidar_on_image

def process_single_frame(frame_id, data_dir, output_dir, max_depth=80.0, show=False):
    # bir karenin lidar ve kamera verilerini birlestirme

    velo_path = os.path.join(data_dir, "velodyne", f"{frame_id}.bin")
    img_path = os.path.join(data_dir, "image_2", f"{frame_id}.jpeg")
    calib_path = os.path.join(data_dir, "calib", f"{frame_id}.txt")
    save_path = os.path.join(output_dir, f"fusion_{frame_id}.png")

    # Modul 1: Veri Yukleme
    raw_pts = load_lidar_points(velo_path)
    image = load_image(img_path)
    calib = load_calibration(calib_path)

    # Modul 2: Projeksiyon ve Filtreleme
    pts_hom = prepare_lidar_points(raw_pts)
    proj_mat = compute_projection_matrix(calib)
    pts_cam = project_to_camera_frame(pts_hom, proj_mat)
    pts_2d, depths = normalize_to_pixel_coords(pts_cam)
    valid_pts_2d, valid_depths, _ = filter_points_in_img(pts_2d, depths, image.shape)

    # Modul 3: Gorsellestirme ve Kaydetme
    plot_lidar_on_image(image, valid_pts_2d, valid_depths, max_depth=max_depth, save_path=save_path)

    if show:
        import matplotlib.pyplot as plt
        plt.show()

def run_batch_pipeline(data_dir, output_dir):
    # Veri setindeki tum kareleri otomatik olarak isler.
    os.makedirs(output_dir, exist_ok=True)
    
    # velodyne klasorundeki tum .bin dosyalarini tara
    bin_files = sorted(glob.glob(os.path.join(data_dir, "velodyne", "*.bin")))
    total_frames = len(bin_files)
    
    print(f"Toplam {total_frames} kare bulundu. Islem basliyor...")

    for idx, bin_path in enumerate(bin_files):
        # dosya adindan ID'yi cek
        frame_id = os.path.splitext(os.path.basename(bin_path))[0]
        print(f"[{idx+1}/{total_frames}] Kare isleniyor: {frame_id}")
        
        process_single_frame(frame_id, data_dir, output_dir, max_depth=60.0, show=False)

    print("Tum veri seti basariyla islendi ve kaydedildi!")

if __name__ == "__main__":
    DATA_DIR = r"C:\Coding\cam_lidar_fusion\kitti_tiny\training"
    OUTPUT_DIR = r"C:\Coding\cam_lidar_fusion\results"

    run_batch_pipeline(DATA_DIR, OUTPUT_DIR)