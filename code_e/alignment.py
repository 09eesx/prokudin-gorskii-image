# alignment.py

import numpy as np
import skimage as sk
import skimage.transform


Green_shift_X = []
Green_shift_Y = []
Red_shift_X = []
Red_shift_Y = []


def NCC(red_channel, green_channel, blue_channel, search):
    """
    Normalleştirilmiş Çapraz Korelasyon (NCC) kullanarak optimal kaymayı bulur.
    """
    # 1. Olası Kaymaları Listeleme
    axis_shift_list = []
    for x_axis in range (-search, search+1):
        for y_axis in range(-search, search+1):
            axis_shift_list.append([x_axis, y_axis])

    # --- YEŞİL KANAL İÇİN NCC ---
    
    green_shift_list = [np.roll(green_channel, shift, axis = (0,1)) for shift in axis_shift_list]
    NCC_scores = []

    # Mavi kanalın normalizasyonu (sabit)
    blue_channel_norm = np.linalg.norm(blue_channel)
    blue_channel_1D = np.ravel(blue_channel)

    for green_shift_image in green_shift_list:
        green_shift_image_norm = np.linalg.norm(green_shift_image)
        green_shift_1D = np.ravel(green_shift_image)
        
        # NCC Hesaplaması: np.dot (normalized vectors)
        NCC_scores.append(np.dot(green_shift_1D / green_shift_image_norm,
                                 blue_channel_1D / blue_channel_norm))
    
    max_index_g = NCC_scores.index(max(NCC_scores))
    green_displacement = axis_shift_list[max_index_g]
    green_shift_final = np.roll(green_channel, green_displacement, axis = (0,1))

    # --- KIRMIZI KANAL İÇİN NCC ---

    red_shift_list = [np.roll(red_channel, shift, axis = (0,1)) for shift in axis_shift_list]
    NCC_scores = []

    for red_shift_image in red_shift_list:
        red_shift_image_norm = np.linalg.norm(red_shift_image)
        red_shift_1D = np.ravel(red_shift_image)
        
        # NCC Hesaplaması: np.dot (normalized vectors)
        NCC_scores.append(np.dot(red_shift_1D / red_shift_image_norm,
                                 blue_channel_1D / blue_channel_norm))
    
    max_index_r = NCC_scores.index(max(NCC_scores))
    red_displacement = axis_shift_list[max_index_r]
    red_shift_final = np.roll(red_channel, red_displacement, axis= (0,1))
    
    # Basit bir NCC için bu printler gereklidir, ancak piramit içinde çağrıldığı için yorum satırı yapılır.
    # print(f"G: {max(NCC_scores):.4f} {green_displacement} | R: {max(NCC_scores):.4f} {red_displacement}")
    
    return green_shift_final, red_shift_final, red_displacement, green_displacement

def Image_pyramid(red_chan, green_chan, blue_chan, depth, search_range):
    """
    Görüntü Piramidi (recursive) kullanarak çok seviyeli hizalama yapar.
    """
    # 0. TEMEL DURUM (Piramidin En Altı: İnce Ayar)
    if depth == 0:
        # Arama aralığını en aza indir
        disp = int(search_range / (2**4)) 
        if disp == 0:
            disp = 1 # Arama aralığı minimum 1 olmalı

        return NCC(red_chan, green_chan, blue_chan, disp)
    
    # 1. YİNELEMELİ ADIM (Piramit Seviyeleri)
    else: 
        
        # Görüntüleri Küçültme (Downsampling)
        red_reduced = sk.transform.rescale(red_chan, 0.5**depth, channel_axis=-1)
        blue_reduced = sk.transform.rescale(blue_chan, 0.5**depth, channel_axis=-1)
        green_reduced = sk.transform.rescale(green_chan, 0.5**depth, channel_axis=-1)
        
        # Arama Alanını Ayarlama (Küçük resimde geniş arama)
        disp = int(search_range / (2**(4-depth)))
        if disp == 0:
            disp = 1
        
        # NCC ile Kaba Kaymayı Bulma
        g_sm, r_sm, r_shft, g_shft = NCC(red_reduced, green_reduced, blue_reduced, disp)
        
        # Kaymayı Ölçeklendirme (Upscaling)
        r_shft[0] = int(r_shft[0]*(2**depth))
        r_shft[1] = int(r_shft[1]*(2**depth))
        
        g_shft[0] = int(g_shft[0]*(2**depth))
        g_shft[1] = int(g_shft[1]*(2**depth))
        
        # Kaymaları Toplama (Global listelere ekleme)
        Green_shift_X.append(g_shft[0])
        Green_shift_Y.append(g_shft[1])
        Red_shift_X.append(r_shft[0])
        Red_shift_Y.append(r_shft[1])
        
        # Kaymayı Uygulama (Tam boyuta roll)
        red_roll = np.roll(red_chan, r_shft, axis=(0,1))
        green_roll = np.roll(green_chan, g_shft, axis = (0,1))
        
        # Bir sonraki seviyeye çağrı (daha ince ayar için)
        return Image_pyramid(red_roll, green_roll, blue_chan, depth-1, search_range)
    
def SSD(red_channel, green_channel, blue_channel, search):
    """
    Kareler Farkının Toplamı (SSD) kullanarak optimal kaymayı bulur.
    SSD en az (minimum) olduğunda en iyi hizalama bulunur.
    """
    
    # 1. Olası Kaymaları Listeleme
    axis_shift_list = []
    for x_axis in range(-search, search + 1):
        for y_axis in range(-search, search + 1):
            axis_shift_list.append([x_axis, y_axis])

    # =========================================================================
    # --- YEŞİL KANAL İÇİN SSD ---
    # =========================================================================
    
    green_shift_list = []
    for shift in axis_shift_list:
        green_shift_list.append(np.roll(green_channel, shift, axis=(0, 1)))

    SSD_scores = []
    
    for green_shift_image in green_shift_list:
        # SSD Hesaplaması: (im1 - im2)**2 toplamı
        # SSD, görüntüler arasındaki parlaklık farklarına hassastır.
        ssd_score = np.sum((green_shift_image - blue_channel) ** 2)
        SSD_scores.append(ssd_score)
        
    # En iyi kayma, minimum SSD skorunu verendir
    min_SSD = min(SSD_scores)
    min_index = SSD_scores.index(min_SSD)

    green_displacement = axis_shift_list[min_index]
    
    print(f"Green Channel SSD Min Skoru: {min_SSD}")
    print(f"Green Channel Shift: {green_displacement}")

    green_shift_final = np.roll(green_channel, green_displacement, axis=(0, 1))

    # =========================================================================
    # --- KIRMIZI KANAL İÇİN SSD ---
    # =========================================================================
    
    red_shift_list = []
    for shift in axis_shift_list:
        red_shift_list.append(np.roll(red_channel, shift, axis=(0, 1)))

    SSD_scores = []

    for red_shift_image in red_shift_list:
        # SSD Hesaplaması
        ssd_score = np.sum((red_shift_image - blue_channel) ** 2)
        SSD_scores.append(ssd_score)

    # En iyi kayma, minimum SSD skorunu verendir
    min_SSD = min(SSD_scores)
    min_index = SSD_scores.index(min_SSD)
    
    red_displacement = axis_shift_list[min_index]
    
    print(f"Red Channel SSD Min Skoru: {min_SSD}")
    print(f"Red Channel Shift: {red_displacement}")

    red_shift_final = np.roll(red_channel, red_displacement, axis=(0, 1))
    
    return (green_shift_final, red_shift_final, red_displacement, green_displacement)