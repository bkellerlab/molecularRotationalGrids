from PIL import Image
import numpy as np
import os
import subprocess


def find_bounding_box(image: Image, threshold: int = 245) -> tuple:
    """
    Find the bounding box of the image, enabling the cutting of unnecessary background.

    Args:
        image (Image): Pillow's Image object
        threshold (int): brightness as int (max 255) values above treshold can be outside the bounding box

    Returns:
        tuple: left, upper, right, and lower pixel coordinate
    """
    img_array = np.array(image)  # Convert image to NumPy array

    # Convert RGB or RGBA to grayscale (brightness)
    if img_array.shape[-1] == 4:  # RGBA image
        r, g, b, a = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2], img_array[:, :, 3]
        brightness = (r * 0.299 + g * 0.587 + b * 0.114) * (a / 255)  # Adjust for transparency
    else:  # RGB image
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        brightness = r * 0.299 + g * 0.587 + b * 0.114  # Standard grayscale conversion

    # Identify non-white pixels
    mask = brightness < threshold
    coords = np.argwhere(mask)

    if coords.size == 0:  # If no non-white pixels are found, return full image
        return None

    # Get bounding box
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return (x_min, y_min, x_max + 1, y_max + 1)  # Adjust for cropping


def get_common_bbox(image_paths, threshold=245):
    """Find the smallest common bounding box for all images."""
    min_x, min_y, max_x, max_y = None, None, None, None

    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")  # Convert to RGB
        bbox = find_bounding_box(img, threshold)


        if bbox:
            x1, y1, x2, y2 = bbox
            if min_x is None or x1 < min_x: min_x = x1
            if min_y is None or y1 < min_y: min_y = y1
            if max_x is None or x2 > max_x: max_x = x2
            if max_y is None or y2 > max_y: max_y = y2

        img.close()

    return (min_x, min_y, max_x, max_y) if min_x is not None else None


def trim_images_with_common_bbox(input_paths, output_paths):
    """Crop all images using the same bounding box and save them."""

    # find the bounding box
    bbox = get_common_bbox(input_paths)
    print(bbox)

    for inp_path, out_path in zip(input_paths, output_paths):
        img = Image.open(inp_path)
        cropped_img = img.crop(bbox)
        cropped_img.save(out_path)
        img.close()

def join_images(input_paths, output_path, flip=True):
    if flip:
        subprocess.run(f"montage -flip {' '.join(input_paths)} {output_path}", shell=True)
    else:
        subprocess.run(f"montage -mode concatenate -tile 4x  {' '.join(input_paths)} {output_path}", shell=True)