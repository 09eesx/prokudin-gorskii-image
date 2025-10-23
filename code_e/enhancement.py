import numpy as np
import cv2

def adjust_gamma(image, gamma=1.1):
    """
    Görüntüye gamma düzeltmesi uygular (parlaklık/kontrast ayarı).
    """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0,256)]).astype("uint8")
    return cv2.LUT(image, table)

def mask_color_range(img, lower_hsv, upper_hsv):
    """
    Görüntüde belirli HSV renk aralığını maskeleyip vurgular.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    return cv2.bitwise_and(img, img, mask=mask)

def enhance_image(img):
    """
    Görüntüyü iyileştirir: kontrast, parlaklık, keskinlik, doygunluk.
    """
    img = img.copy()

    # LAB uzayında CLAHE (parlaklık iyileştirme)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge([l,a,b])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Gamma düzeltmesi
    img = adjust_gamma(img, gamma=1.1)

    # Laplasyen keskinleştirme
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    img = cv2.filter2D(img, -1, kernel)

    # HSV uzayında doygunluk artırma
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.equalizeHist(s)
    hsv = cv2.merge([h, s, v])
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Normalizasyon (0-255)
    out = np.empty_like(img)  # img ile aynı shape/dtype
    cv2.normalize(img, out, 0, 255, cv2.NORM_MINMAX)
    img = out

    return img

# Örnek kullanım:
# from enhancement import enhance_image
# improved = enhance_image(image)
