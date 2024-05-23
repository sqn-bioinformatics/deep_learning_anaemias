import os
import cv2
import hashlib
import numpy as np
from skimage.segmentation import clear_border
import pickle


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

    images = os.listdir(path_in)
    
    my_dict = {}

    
    for img_name in images:
        img_path = os.path.join(path_in, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: {img_path} could not be read, skipping.")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 3)
        edges = cv2.Canny(blur, threshold1=150, threshold2=200)
        no_bord = clear_border(edges)

        if is_image_empty(no_bord):
            print(f"Image {img_name} is empty after edge detection, skipping.")
            continue

        contours, hierarchy = cv2.findContours(
            edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        valid_contours = []

        for contour, h in zip(contours, hierarchy[0]):
            # Check if the contour is not inside another contour (parent index is -1)
            if h[3] == -1:
                area = cv2.contourArea(contour)
                # Check if the area is less than 1200 pixels
                if 200 <= area <= 1000:
                    valid_contours.append(contour)

        if len(valid_contours) != 1:
            print(f"Image {img_name} does not have exactly one cell, skipping.")
            continue

        hash_name = get_name(gray)        
        output_path = os.path.join(path_out, f"{hash_name}.jpeg")

        cv2.imwrite(output_path, gray)
        print(f"Processed and saved {img_name} as {hash_name}.jpeg")
        
        my_dict.setdefault(hash_name, [])
        my_dict[hash_name].append(img_path)
        
        with open('filename.pickle', 'wb') as handle: # start here https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict-or-any-other-python-object
            pickle.dump(my_dict, handle, protocol=pickle.HIGH
        
    

if __name__ == "__main__":
    main()
