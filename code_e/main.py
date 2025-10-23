import time
import numpy as np
import skimage.io as skio
import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from enhancement import enhance_image

from code_e.utils_e import goruntu_hizalama, save_and_display_results
from code_e.alignment import Image_pyramid


Green_shift_X = []
Green_shift_Y = []
Red_shift_X = []
Red_shift_Y = []

def run_alignment_pipeline(search_range, image_name, crop_percent, pyramid_depth, img_path):
    """
    Ana hizalama ve renklendirme boru hattını çalıştırır.
    Args:
        search_range (int): Hizalama için arama aralığı.
        image_name (str): Renkli görüntü için dosya adı.
        crop_percent (float): Görüntünün kırpılma yüzdesi.
        pyramid_depth (int): Görüntü piramidinin derinliği.
        img_path (str): Giriş görüntüsünün yolu.
    """
    global Green_shift_X, Green_shift_Y, Red_shift_X, Red_shift_Y

    start_time = time.time()

    print(f"--- {image_name} için hizalama başlatılıyor ---")
    image_Z = skio.imread(img_path)

    red_channel, blue_channel, green_channel = goruntu_hizalama(image_Z, crop_percent)

    print("Hizalama için görüntü piramidi başlatılıyor...")
    green_shift_final, red_shift_final, red_displacement, green_displacement = \
        Image_pyramid(red_channel, green_channel, blue_channel, pyramid_depth, search_range)
    
    im_out = np.dstack((red_shift_final, green_shift_final, blue_channel))

    total_time = time.time() - start_time

    total_g_shift = [sum(Green_shift_X), sum(Green_shift_Y)]
    total_r_shift = [sum(Red_shift_X), sum(Red_shift_Y)]

    save_and_display_results(im_out, image_name, search_range, total_time, total_r_shift, total_g_shift)


def process_single_image(file_path, search=15, crop=5, depth=4):
    """
    Tek bir görüntüyü hizala, iyileştir ve kaydet.
    """
    name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"Başladı -> {name}")

    start = time.time()
    image_Z = skio.imread(file_path)
    red, blue, green = goruntu_hizalama(image_Z, crop)

    g_final, r_final, r_disp, g_disp = Image_pyramid(red, green, blue, depth, search)
    im_out = np.dstack((r_final, g_final, blue))
    im_out = enhance_image(im_out)

    total_time = time.time() - start
    save_and_display_results(im_out, name, search, total_time, r_disp, g_disp)
    print(f"Bitti -> {name} | Süre: {total_time:.2f} sn")
    return name


def parallel_batch_process(data_folder, workers=4):
    """
    Klasördeki tüm .jpg dosyalarını paralel olarak işler.
    """
    files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.lower().endswith(".jpg")]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_single_image, f): f for f in files}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as e:
                print(f"Hata: {e}")

if __name__ == "__main__":
    data_folder = "data_e"
    parallel_batch_process(data_folder, workers=8) 