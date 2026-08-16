import numpy as np
import cv2 as cv

def img_crop(img, size = 768, resize = 512):
    green = img[:,:,1]
    blur = cv.GaussianBlur(green, (75, 75), 0)

    #center finding
    _, _, _, max_loc = cv.minMaxLoc(blur)

    #padding
    half = size // 2
    padded = cv.copyMakeBorder(img, half, half, half, half, borderType= cv.BORDER_CONSTANT, value= [0, 0, 0])

    #bounding box coordinates
    x, y = np.array(max_loc) + half
    x0, y0 = x - half, y - half
    x1, y1 = x + half, y + half

    #cropping & resizing
    cropped = padded[y0:y1, x0:x1].copy()
    resized = cv.resize(cropped, (resize, resize), interpolation = cv.INTER_LINEAR)

    return resized, (x0, y0, x1, y1)

def mask_crop(mask, loc, size=768, resize=512):
    x0, y0, x1, y1 = loc

    #padding
    half = size // 2
    padded = cv.copyMakeBorder(mask, half, half, half, half, borderType= cv.BORDER_CONSTANT, value= 255)

    #cropping & resizing
    cropped = padded[y0:y1, x0:x1].copy()
    resized = cv.resize(cropped, (resize, resize), interpolation = cv.INTER_NEAREST)

    return resized