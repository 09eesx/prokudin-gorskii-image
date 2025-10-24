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
#  BONUS: OTOMATİK KENAR KIRPMA (10 PUAN)
# ============================================================
def auto_border_crop(img, threshold_ratio=0.10, min_crop_pixels=5):
    """
     Otomatik Kenar Kırpma Fonksiyonu
    Görüntü kenarlarındaki siyah çerçeveleri ve hizalama artifaktlarını
    otomatik olarak tespit edip kırpar.

    Önerilen analiz:
    - Kenar piksellerinin yoğunluk ortalamasını hesapla
    - Eşik değerin altındaki bölgeleri kırp
    - Dört kenarda ayrı ayrı kırpma uygula

    Args:
        img (ndarray): Giriş görüntüsü (RGB veya grayscale)
        threshold_ratio (float): 0-1 arası kırpma eşiği oranı (daha yüksek = agresif kırpma)
        min_crop_pixels (int): Minimum kırpılacak piksel miktarı (her kenar)

    Returns:
        cropped_img (ndarray): Kırpılmış yeni görüntü
    """
    import numpy as np

    # Griye dönüştür
    if img.ndim == 3:
        gray = np.mean(img, axis=2)
    else:
        gray = img.copy()

    # Normalize et (0–1 arası)
    gray = gray.astype(np.float32)
    gray /= np.max(gray) if np.max(gray) != 0 else 1

    h, w = gray.shape
    threshold = threshold_ratio * np.mean(gray)

    # Kenar yoğunluklarını analiz et
    top, bottom, left, right = 0, h, 0, w

    # Üst kenar
    for i in range(h // 4):
        if np.mean(gray[i, :]) < threshold:
            top = i + min_crop_pixels
        else:
            break

    # Alt kenar
    for i in range(h - 1, h * 3 // 4, -1):
        if np.mean(gray[i, :]) < threshold:
            bottom = i - min_crop_pixels
        else:
            break

    # Sol kenar
    for j in range(w // 4):
        if np.mean(gray[:, j]) < threshold:
            left = j + min_crop_pixels
        else:
            break

    # Sağ kenar
    for j in range(w - 1, w * 3 // 4, -1):
        if np.mean(gray[:, j]) < threshold:
            right = j - min_crop_pixels
        else:
            break

    # Limitleri güvenli hale getir
    top = max(0, top)
    left = max(0, left)
    bottom = min(h, bottom)
    right = min(w, right)

    # Kırpılmış görüntüyü döndür
    cropped = img[top:bottom, left:right]
    print(f" Otomatik kırpma uygulandı -> Yeni boyut: {cropped.shape}")
    return cropped