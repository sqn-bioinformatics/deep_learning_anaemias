import os
import cv2
import hashlib
import numpy as np
import pickle
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from skimage.segmentation import clear_border
from tqdm import tqdm

logger = logging.getLogger(__name__)

def is_image_empty(img):  # to check if cells are present
    return np.all(img == 0)


def get_name(img):
    bytes = img.tobytes()
    salt = os.urandom(16)  # 16 bytes of randomness
    bytes_with_salt = bytes + salt  # append the salt to the file content
    readable_hash = hashlib.sha256(bytes_with_salt).hexdigest()
    return readable_hash[:32]


def image_rotate_crop(src, cnt, size):
    if len(src.shape) != 2:
        raise ValueError(
            "image_rotate_crop(): Source image should be in grayscale,",
            "and have two dimensions.")
    image_width, image_height = src.shape  # cv2 takes rows and cols swapped 
    M = cv2.moments(cnt)
    # Calculate x,y coordinate of center
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    (_, (rect_width, rect_height), angle) = cv2.minAreaRect(cnt)
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1)
    
    img_rot = cv2.warpAffine(
        src, M, (image_width, image_height),flags=cv2.INTER_LINEAR
        )

    # Calculate the top-left and bottom-right coordinates
    half_size = size // 2
    x1, y1 = max(0, cx - half_size), max(0, cy - half_size)
    x2, y2 = min(image_width, cx + half_size), min(image_height, cy + half_size)

    result = img_rot[y1:y2, x1:x2]

    # Make long side up
    if rect_height < rect_width:
        return cv2.rotate(result, cv2.ROTATE_90_CLOCKWISE)
    else:
        return result


def main():
    file_path = os.path.dirname(__file__)
    path_in = "/home/t.afanasyeva/deep_learning_anaemias/resources/images_subset"
    path_out = "/home/t.afanasyeva/deep_learning_anaemias/resources/out"
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    logging.basicConfig(
        filename=os.path.join(file_path, f'{timestamp}converter.log'),
        level=logging.INFO
        )
    logger.info('Started')
 
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    images = os.listdir(path_in)

    dict_hash_old_name = {}
    area_list = []

    for img_name in tqdm(images[:1000]):
        img_path = os.path.join(path_in, img_name)
        img = cv2.imread(img_path)

        if img is None:
            logger.info(f"Warning: {img_path} could not be read, skipping.")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 3)
        edges = cv2.Canny(blur, threshold1=150, threshold2=200)
        no_bord = clear_border(edges)

        if is_image_empty(no_bord):
            logger.info(f"Image {img_name} is empty after edge detection, skipping.")
            continue

        contours, hierarchy = cv2.findContours(
            edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        valid_contours = []

        for contour, h in zip(contours, hierarchy[0]):
            # Check if the contour is not inside another contour (parent index is -1)
            if h[3] == -1:
                area = cv2.contourArea(contour)
                area_list.append(area)
                # Check if the area is less than 1000 pixels
                if 250 <= area <= 1000:
                    valid_contours.append(contour)

        if len(valid_contours) != 1:
            logger.info(f"Image {img_name} does not have exactly one cell, skipping.")
            continue
        
        new_image_size = 64
        rotated_croped = image_rotate_crop(gray, valid_contours[0], new_image_size)
        
        if rotated_croped.shape[0] != new_image_size:
            logger.info(f"Image {img_name} contains cell too close to the boarder.")
            continue
            
        hash_name = get_name(rotated_croped)
        output_path = os.path.join(path_out, f"{hash_name}.jpeg")

        cv2.imwrite(output_path, rotated_croped)
        logger.info(f"Processed and saved {img_name} as {hash_name}.jpeg")
        dict_hash_old_name.setdefault(hash_name, [])
        dict_hash_old_name[hash_name].append(img_path)

    logger.info(
        f"Successfully processed {len(dict_hash_old_name.keys())/len(images):.2f} percent of all images."
          )

    # Distribution of areas surfaces
    plt.hist(area_list, bins=5, edgecolor='black')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Histogram of Values')
    plt.savefig(os.path.join(file_path, "areas.png"))


    # pickle.dump(dict_hash_old_name, handle, protocol=pickle.HIGH)


if __name__ == "__main__":
    # main()
    import cProfile
    import pstats

    prof = cProfile.Profile()
    prof.enable()
    main()
    prof.disable()
    
    prof.dump_stats('output.prof')
    
    with open(os.path.join(os.path.dirname(__file__), 'performance.log'), 'w') as stream:
        stats = pstats.Stats('output.prof', stream=stream)
        stats.sort_stats('cumtime')
        stats.print_stats()

