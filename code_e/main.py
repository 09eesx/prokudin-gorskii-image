import time
import numpy as np
import skimage.io as skio
import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    search = 35
    name = input("Lütfen renkli görüntü için dosya adını giriniz (uzantısız): ")
    crop_amount = 4
    depth = 10
    image_path = input("Lütfen hizalanacak görüntünün yolunu giriniz: ")

    run_alignment_pipeline(search, name, crop_amount, depth, image_path)

