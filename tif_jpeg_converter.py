import os
import cv2
import hashlib
import numpy as np


def is_image_empty(img):  # to check if cells are present
    return np.all(img == 0)


def get_name(img):
    bytes = img.tobytes()
    salt = os.urandom(16)  # 16 bytes of randomness
    bytes_with_salt = bytes + salt  # append the salt to the file content
    readable_hash = hashlib.sha256(bytes_with_salt).hexdigest()
    return readable_hash[:32]


def main():
    path_in = "/home/t.afanasyeva/deep_learning_anaemias/resources/images_subset"
    path_out = "/home/t.afanasyeva/deep_learning_anaemias/resources/out"

    if not os.path.exists(path_out):
        os.makedirs(path_out)
    images = os.path.listdir(path_in)
    print(images)

    for img_name in images:
        img_path = os.path.join(path_in, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Warning: {img_path} could not be read, skipping.")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 3)
        edges = cv2.Canny(blur, threshold1=150, threshold2=200)

        if is_image_empty(edges):
          print(f"Image {img_name} is empty after edge detection, skipping.")
          continue  

        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        valid_contours = []

        for contour, h in zip(contours, hierarchy[0]):
        # Check if the contour is not inside another contour (parent index is -1)
        if h[3] == -1:
        area = cv2.contourArea(contour)
        # Check if the area is less than 1200 pixels
        if if 200 <= area <= 1200:
        valid_contours.append(contour)

        if len(valid_contours) != 1:
            print(f"Image {img_name} does not have exactly one cell, skipping.")
            continue
      
        hash_name = get_name(gray)
        output_path = os.path.join(path_out, f"{hash_name}.jpeg")
        cv2.imwrite(output_path, gray)
        print(f"Processed and saved {img_name} as {hash_name}.jpeg")
       
if __name__ == "__main__":
    main()
