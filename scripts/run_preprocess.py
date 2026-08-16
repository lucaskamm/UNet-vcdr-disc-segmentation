import os
import cv2 as cv
from src.preprocess import img_crop, mask_crop

raw_dir = "data/raw/REFUGE2"
processed_dir = "data/processed/refuge2-disc-processed"

splits = ["train", "val", "test"]

for split in splits:
    #img and mask directories
    raw_img_dir = os.path.join(raw_dir, split, "images")
    raw_mask_dir = os.path.join(raw_dir, split, "mask")
    processed_img_dir = os.path.join(processed_dir, split, "images")
    processed_mask_dir = os.path.join(processed_dir, split, "mask")

    #Create processed directories if they don't exist
    os.makedirs(processed_img_dir, exist_ok=True)
    os.makedirs(processed_mask_dir, exist_ok=True)

    i = 0
    mask_ext = ".png" if split == "val" else ".bmp"

    for img_name in os.listdir(raw_img_dir):
        #path
        name, _ = os.path.splitext(img_name)
        img_path = os.path.join(raw_img_dir, img_name)
        mask_path = os.path.join(raw_mask_dir, name + mask_ext)

        #read image and mask
        img = cv.imread(img_path)
        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)

        #process image and mask
        cropped_img, loc = img_crop(img)
        cropped_mask = mask_crop(mask, loc)

        #save processed image and mask
        processed_img_path = os.path.join(processed_img_dir, name + ".png")
        processed_mask_path = os.path.join(processed_mask_dir, name + ".png")
        cv.imwrite(processed_img_path, cropped_img)
        cv.imwrite(processed_mask_path, cropped_mask)

        i += 1

    print(f"======PROCESSED======\n------{split}------\nImages: {i} \nMasks: {i}", end = '\n\n')