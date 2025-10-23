import numpy as np
import skimage as sk
import skimage.io as skio
import matplotlib.pyplot as plt
import os

Green_shift_X = []
Green_shift_Y = []
Red_shift_X = []
Red_shift_Y = []

def goruntu_hizalama(image_Z, crop_amount):
    """
    Renkli görüntüyü kırpar ve renk kanallarını ayırır.
    Args:
        image_Z (ndarray): Giriş renkli görüntü.
        crop_amount (float): Görüntünün kırpılma yüzdesi.
    Returns:
        red_channel (ndarray): Kırpılmış kırmızı kanal.
        blue_channel (ndarray): Kırpılmış mavi kanal.
        green_channel (ndarray): Kırpılmış yeşil kanal.
    """
    original_height = image_Z.shape[0]
    original_width = image_Z.shape[1]

    reduc_height = int(original_height / 3)
    reduc_width = int(original_width)

    blue_channel = image_Z[:reduc_height]
    green_channel = image_Z[reduc_height: 2 * reduc_height]
    red_channel = image_Z[2 * reduc_height: 3 * reduc_height]

    crop_height = int(reduc_height * (crop_amount / 100))
    crop_width = int(reduc_width * (crop_amount / 100))

    left_crop = crop_width
    right_crop = reduc_width - crop_width
    top_crop = crop_height
    bottom_crop = reduc_height - crop_height

    red_channel = red_channel[top_crop:bottom_crop, left_crop:right_crop]
    blue_channel = blue_channel[top_crop:bottom_crop, left_crop:right_crop]
    green_channel = green_channel[top_crop:bottom_crop, left_crop:right_crop]

    return red_channel, blue_channel, green_channel

def save_and_display_results(im_out, name, search, total_time, total_r_shift, total_g_shift):
    """
    Hizalanmış görüntüyü kaydeder ve sonuçları görüntüler.
    Args:
        im_out (ndarray): Hizalanmış renkli görüntü.
        name (str): Görüntü adı.
        total_time (float): Toplam hizalama süresi.
        total_r_shift (list): Kırmızı kanal için toplam kayma.
        total_g_shift (list): Yeşil kanal için toplam kayma.
    """

    # 16-bit (uint16) veriyi 8-bit (uint8) aralığına dönüştürme (görselleştirme için)
    im_out_norm = im_out / np.max(im_out) 
    im_out_8bit = (im_out_norm * 255).astype(np.uint8) 

    # Kaydetme
    fname = f'results/{name}_NCC_Pyramid_{search}.jpg'
    # Klasör yoksa oluştur (Profesyonel yaklaşım)
    os.makedirs('results', exist_ok=True) 
    skio.imsave(fname, im_out_8bit) 

    # Konsol Çıktısı
    print("\n--- SONUÇLAR ---")
    print(f'Total time = {total_time:.4f} seconds')
    print(f'Red channel total shift = {total_r_shift}')
    print(f'Green channel total shift = {total_g_shift}')

    # Görselleştirme
    plt.figure(figsize = (10,10))
    skio.imshow(fname)
    plt.title(f'Final Aligned Image (Total Time: {total_time:.2f}s)')
    plt.axis('off')
    plt.show()