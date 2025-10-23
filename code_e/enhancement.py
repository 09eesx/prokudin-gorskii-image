# enhancement.py

import numpy as np
import cv2

def adjust_gamma(image, gamma=1.0):
    """
    Görüntüye Gamma düzeltme uygular (Parlaklık ve kontrast ayarlama).
    """
    invGamma = 1.0 / gamma
    # Piksel değerlerini dönüştürmek için bir arama tablosu (LUT) oluşturur.
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    
    # Dönüşümü uygular
    return cv2.LUT(image, table)

def mask_color_range(img, lower_hsv, upper_hsv):
    """
    Görüntüde belirli bir HSV aralığındaki rengi maskeler ve vurgular.
    """
    # cv2'nin BGR formatında çalıştığını varsayarız.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Maske oluşturma
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    # Maskeyi orijinal görüntüye uygulama
    result = cv2.bitwise_and(img, img, mask=mask)
    return result