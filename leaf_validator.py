import cv2
import numpy as np


def validate_leaf(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return False, "Invalid image"


    # -----------------------------------------
    # Resize image
    # -----------------------------------------

    image = cv2.resize(image, (224, 224))


    # -----------------------------------------
    # Convert to HSV
    # -----------------------------------------

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


    # -----------------------------------------
    # Green color detection
    # -----------------------------------------

    lower_green = np.array([25, 30, 30])
    upper_green = np.array([95, 255, 255])

    green_mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )


    # -----------------------------------------
    # Calculate green percentage
    # -----------------------------------------

    green_pixels = cv2.countNonZero(green_mask)

    total_pixels = image.shape[0] * image.shape[1]

    green_ratio = green_pixels / total_pixels


    print(
        f"GREEN RATIO = {green_ratio:.3f}",
        flush=True
    )


    # -----------------------------------------
    # Basic leaf check
    # -----------------------------------------

    if green_ratio < 0.08:

        return False, (
            "This does not appear to be a plant leaf. "
            "Please upload a clear Tomato, Potato, "
            "or Bell Pepper leaf image."
        )


    # -----------------------------------------
    # Find contours
    # -----------------------------------------

    contours, _ = cv2.findContours(
        green_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    if not contours:

        return False, (
            "No leaf-like object detected. "
            "Please upload a clear plant leaf image."
        )


    # -----------------------------------------
    # Largest green object
    # -----------------------------------------

    largest_contour = max(
        contours,
        key=cv2.contourArea
    )


    area = cv2.contourArea(
        largest_contour
    )


    image_area = image.shape[0] * image.shape[1]

    area_ratio = area / image_area


    print(
        f"LEAF AREA RATIO = {area_ratio:.3f}",
        flush=True
    )


    if area_ratio < 0.03:

        return False, (
            "A clear leaf could not be detected. "
            "Please upload a Tomato, Potato, "
            "or Bell Pepper leaf."
        )


    return True, "Leaf detected"
