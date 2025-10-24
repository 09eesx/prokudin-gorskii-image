import time
import numpy as np
import skimage.io as skio
import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from code_e.enhancement import enhance_image

from concurrent.futures import ThreadPoolExecutor, as_completed

from code_e.utils_e import goruntu_hizalama, save_and_display_results, auto_border_crop
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


def process_single_image(image_path):
    """ Tek bir görüntü üzerinde hizalama + otomatik kırpma işlemi yapar. """
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    print(f"\n--- {image_name} için hizalama başlatılıyor ---")

    start_time = time.time()
    image_Z = skio.imread(image_path)
    search, crop, depth = 15, 5, 4  # Varsayılan parametreler

    red_channel, blue_channel, green_channel = goruntu_hizalama(image_Z, crop)

    green_shift_final, red_shift_final, red_displacement, green_displacement = \
        Image_pyramid(red_channel, green_channel, blue_channel, depth, search)

    im_out = np.dstack((red_shift_final, green_shift_final, blue_channel))

    # BONUS: otomatik kenar kırpma
    im_out = auto_border_crop(im_out)

    total_time = time.time() - start_time
    total_g_shift = [sum(Green_shift_X), sum(Green_shift_Y)]
    total_r_shift = [sum(Red_shift_X), sum(Red_shift_Y)]

    save_and_display_results(im_out, image_name, search, total_time, total_r_shift, total_g_shift)
    print(f" {image_name} tamamlandı! ({total_time:.2f} sn)")
    return image_name


# ============================================================
#  PARALEL ÇALIŞTIRMA BLOĞU
# ============================================================
if __name__ == "__main__":
    folder_path = input(" Lütfen hizalanacak görüntülerin bulunduğu klasörü giriniz: ").strip()
    supported_exts = ('.tif', '.tiff', '.jpg', '.jpeg', '.png')
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                    if f.lower().endswith(supported_exts)]

    if not image_files:
        print(" Klasörde hizalanacak uygun görüntü bulunamadı.")
        exit(0)

    print(f"\n {len(image_files)} görüntü bulundu. Paralel işlem başlıyor...\n")

    #  CPU çekirdeği kadar worker aç
    max_workers = min(4, os.cpu_count() or 2)  # çok çekirdekli sistemde 4 worker sınırı
    print(f" Kullanılan worker sayısı: {max_workers}\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_image, path): path for path in image_files}

        for future in as_completed(futures):
            try:
                result = future.result()
                print(f" {result} başarıyla işlendi.")
            except Exception as e:
                print(f" Hata oluştu: {e}")

print("\n Tüm görüntüler paralel olarak işlendi!")