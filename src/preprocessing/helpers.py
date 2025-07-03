import cv2
import numpy as np
from PIL import Image


def crop_yolo_subpix(img: np.ndarray, x_c: float, y_c: float, w: float, h: float) -> np.ndarray:
    H, W = img.shape[:2]
    patch_w = w * W
    patch_h = h * H
    center = (x_c * W, y_c * H)
    size = (int(round(patch_w)), int(round(patch_h)))
    return cv2.getRectSubPix(img, patchSize=size, center=center)


def sharpen_image(img_pil):
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(img_cv, -1, kernel)
    img_sharp = Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
    return img_sharp
