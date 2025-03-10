import os
import cv2
import hashlib
import numpy as np
import pickle
import matplotlib.pyplot as plt
import logging
import argparse
from datetime import datetime
from tqdm import tqdm
from importlib import reload
import pathlib


reload(logging)
logger = logging.getLogger(__name__)


def is_image_empty(img):  # Check if cells are present
    return np.all(img == 0)


def get_name(img):
    bytes = img.tobytes()
    salt = os.urandom(16)  # 16 bytes of randomness
    bytes_with_salt = bytes + salt  # Append the salt to the file content
    readable_hash = hashlib.sha256(bytes_with_salt).hexdigest()
    return readable_hash[:32]


def image_rotate_crop(src, cnt, size):
    if len(src.shape) != 2:
        raise ValueError(
            "image_rotate_crop(): Source image should be in grayscale,",
            "and have two dimensions.",
        )
    image_width, image_height = src.shape  # cv2 takes rows and cols swapped
    M = cv2.moments(cnt)

    cx = int(M["m10"] / M["m00"])  # Calculate x,y coordinate of center
    cy = int(M["m01"] / M["m00"])

    # Get box around the contour
    (_, (rect_width, rect_height), angle) = cv2.minAreaRect(cnt)
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1)
    img_rot = cv2.warpAffine(
        src, M, (image_width, image_height), flags=cv2.INTER_LINEAR
    )

    # Calculate the top-left and bottom-right coordinates
    half_size = size // 2
    x1, y1 = max(0, cx - half_size), max(0, cy - half_size)
    x2, y2 = min(image_width, cx + half_size), min(image_height, cy + half_size)
    img_rot_crop = img_rot[y1:y2, x1:x2]

    # Make long side up
    if rect_height < rect_width:
        return cv2.rotate(img_rot_crop, cv2.ROTATE_90_CLOCKWISE)
    else:
        return img_rot_crop


def check_border_pixels(src):
    rows, cols = src.shape
    border_coordinates = []

    top_row = [(0, col) for col in range(cols)]
    bottom_row = [(rows - 1, col) for col in range(cols)]
    # Left and right columns exclude corners
    left_col = [(row, 0) for row in range(1, rows - 1)]
    right_col = [(row, cols - 1) for row in range(1, rows - 1)]

    border_coordinates.extend(top_row)
    border_coordinates.extend(bottom_row)
    border_coordinates.extend(left_col)
    border_coordinates.extend(right_col)

    threshold = 150
    saturated_pixels_count = 0

    for pixel in border_coordinates:
        if src[pixel[0]][pixel[1]] < threshold:
            saturated_pixels_count += 1

    if saturated_pixels_count >= 5:
        return True
    else:
        return False


def make_graph_areas(area_list, path_out_logs, timestamp):
    # Distribution of areas surfaces
    plt.hist(area_list, bins=20, edgecolor="black")
    plt.xlabel("pixels")
    plt.ylabel("Number of contours")
    plt.title("Histogram of area surface sizes")
    plt.savefig(os.path.join(path_out_logs, f"{timestamp}areas.png"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--files",
        help="path to images to convert to png",
        type=pathlib.Path,
        required=True,
    )
    args = parser.parse_args()
    PATH_TO_FILES = args.files
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    path_out_logs = os.path.join(parent_dir, "logs")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    logging.basicConfig(
        filename=os.path.join(path_out_logs, f"{timestamp}_temp_converter.log"),
        level=logging.INFO,
    )
    logger.info("Started")

    dict_hash_old_name = {}  # Keep track of old file names
    area_list = []  # Keep track of cell sizes fot filtering

    paths = [path[0] for path in os.walk(PATH_TO_FILES)]

    for path_in in paths:
        path_out_folder_name = path_in.split("/")[-1]
        path_out = os.path.join(
            parent_dir, "resources/out/cytpix/", str(path_out_folder_name)
        )
        if not os.path.exists(path_out):
            os.makedirs(path_out)
        images = os.listdir(path_in)

        for img_name in tqdm(images):
            img_path = os.path.join(path_in, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            if img is None:
                logger.info(f"Warning: {img_path} could not be read, skipping.")
                continue
            gray = cv2.normalize(
                img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )

            ext_border = cv2.copyMakeBorder(
                gray,
                top=64,
                bottom=64,
                left=64,
                right=64,
                borderType=cv2.BORDER_REPLICATE,
            )
            blur = cv2.GaussianBlur(ext_border, (5, 5), 3)
            _, thresh = cv2.threshold(
                blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            contours, hierarchy = cv2.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
            )

            valid_contours = []
            try:
                for contour, h in zip(contours, hierarchy[0]):
                    # Check if the contour is not inside another (parent index is -1)
                    if h[3] == -1:
                        area = cv2.contourArea(contour)
                        area_list.append(area)
                        if 200 <= area <= 900:
                            valid_contours.append(contour)
            except TypeError as e:
                logger.info(f"TypeError occurred: {e}")
                continue

            if len(valid_contours) != 1:
                logger.info(
                    f"Image {img_name} does not have exactly one cell, skipping."
                )
                continue

            new_image_size = 64
            ext_border = cv2.cvtColor(ext_border, cv2.COLOR_BGR2GRAY)
            rotated_cropped = image_rotate_crop(
                ext_border, valid_contours[0], new_image_size
            )

            if rotated_cropped.shape != (new_image_size, new_image_size):
                logger.info(f"Image {img_name} contains cell too close to the boarder.")
                continue

            if check_border_pixels(rotated_cropped) is True:
                continue

            hash_name = get_name(rotated_cropped)
            output_path = os.path.join(path_out, f"{hash_name}.png")

            cv2.imwrite(output_path, rotated_cropped)
            dict_hash_old_name.setdefault(hash_name, [])
            dict_hash_old_name[hash_name].append(img_path)

            logger.info(f"Processed and saved {img_name} as {hash_name}.png")

    area_list = [i for i in area_list if i < 1200]
    make_graph_areas(area_list, path_out_logs, timestamp)

    logger.info(
        f"Successfully processed {(len(dict_hash_old_name.keys())/len(images)*100):.2f} percent of all images."
    )
    logger.info(f"Final number of images is {len(dict_hash_old_name.keys())}).")

    with open(
        os.path.join(path_out_logs, f"{timestamp}dict_hash_old_name.pkl"), "wb"
    ) as handle:
        pickle.dump(dict_hash_old_name, handle)


if __name__ == "__main__":
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path_out_logs = os.path.join(parent_dir, "logs")

    if not os.path.exists(path_out_logs):
        os.makedirs(path_out_logs)

    import cProfile
    import pstats

    prof = cProfile.Profile()
    prof.enable()

    main()
    prof.disable()

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    prof.dump_stats(os.path.join(path_out_logs, "output.prof"))
    with open(
        os.path.join(path_out_logs, f"{timestamp}performance.log"), "w"
    ) as stream:
        stats = pstats.Stats(os.path.join(path_out_logs, "output.prof"), stream=stream)
        stats.sort_stats("cumtime")
        stats.print_stats()
