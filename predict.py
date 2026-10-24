import tensorflow as tf

# Limit TensorFlow threads for deployment
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model


print("Loading AgroVision AI model...", flush=True)

model = load_model(
    "model/agrovision_best_model.h5",
    compile=False
)

print("Model loaded successfully", flush=True)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Pepper Bell Bacterial Spot",
    "Pepper Bell Healthy",
    "Potato Early Blight",
    "Potato Late Blight",
    "Potato Healthy",
    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Septoria Leaf Spot",
    "Tomato Spider Mites",
    "Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato Mosaic Virus",
    "Tomato Healthy"
]


class PredictionFailure(str):

    def __new__(cls, message, category):
        failure = super().__new__(cls, message)
        failure.category = category
        return failure


# ============================================================
# LEAF VALIDATION
# ============================================================

def validate_leaf_image(image_path):

    try:

        print("VALIDATING INPUT IMAGE...", flush=True)

        img = cv2.imread(image_path)

        if img is None:

            return False, PredictionFailure(
                "Unable to read this image. "
                "Please upload a clear image of a tomato, "
                "potato, or bell pepper leaf.",
                "invalid_image"
            )

        # ----------------------------------------------------
        # IMAGE SIZE
        # ----------------------------------------------------

        height, width = img.shape[:2]

        print(
            f"VALIDATION IMAGE SIZE: {width} x {height}",
            flush=True
        )

        if width < 100 or height < 100:

            return False, PredictionFailure(
                "The uploaded image is too small. "
                "Please upload a clear image of a tomato, "
                "potato, or bell pepper leaf.",
                "invalid_image"
            )

        # ----------------------------------------------------
        # RESIZE FOR VALIDATION
        # ----------------------------------------------------

        small = cv2.resize(
            img,
            (224, 224)
        )

        # ----------------------------------------------------
        # CONVERT TO HSV
        # ----------------------------------------------------

        hsv = cv2.cvtColor(
            small,
            cv2.COLOR_BGR2HSV
        )

        # ----------------------------------------------------
        # GREEN / PLANT-LIKE PIXELS
        # ----------------------------------------------------

        lower_green = np.array(
            [25, 30, 20]
        )

        upper_green = np.array(
            [100, 255, 255]
        )

        green_mask = cv2.inRange(
            hsv,
            lower_green,
            upper_green
        )

        green_ratio = (
            np.count_nonzero(green_mask)
            /
            green_mask.size
        )

        print(
            f"GREEN RATIO: {green_ratio:.3f}",
            flush=True
        )

        # ----------------------------------------------------
        # SATURATION CHECK
        # ----------------------------------------------------

        saturation = hsv[:, :, 1]

        saturation_ratio = (
            np.mean(saturation > 30)
        )

        print(
            f"SATURATION RATIO: {saturation_ratio:.3f}",
            flush=True
        )

        # ----------------------------------------------------
        # EDGE / TEXTURE CHECK
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            small,
            cv2.COLOR_BGR2GRAY
        )

        edges = cv2.Canny(
            gray,
            50,
            150
        )

        edge_ratio = (
            np.count_nonzero(edges)
            /
            edges.size
        )

        print(
            f"EDGE RATIO: {edge_ratio:.3f}",
            flush=True
        )

        # ----------------------------------------------------
        # LEAF-LIKE SCORE
        # ----------------------------------------------------

        score = 0

        # Green vegetation is a strong indicator
        if green_ratio >= 0.15:
            score += 2

        if green_ratio >= 0.30:
            score += 1

        # Natural plant images usually contain some
        # color/saturation variation
        if saturation_ratio >= 0.20:
            score += 1

        # Leaf images usually contain visible structure
        if edge_ratio >= 0.03:
            score += 1

        print(
            f"LEAF VALIDATION SCORE: {score}/5",
            flush=True
        )

        # ----------------------------------------------------
        # FINAL VALIDATION
        # ----------------------------------------------------

        if score < 3:

            print(
                "IMAGE REJECTED: NOT LEAF-LIKE",
                flush=True
            )

            return False, (
                "This does not appear to be a plant leaf. "
                "Please upload a clear image of a tomato, "
                "potato, or bell pepper leaf."
            )

        print(
            "IMAGE PASSED LEAF VALIDATION",
            flush=True
        )

        return True, None

    except Exception as e:

        print(
            f"LEAF VALIDATION ERROR: {str(e)}",
            flush=True
        )

        return False, PredictionFailure(
            "Unable to validate this image. "
            "Please upload a clear image of a tomato, "
            "potato, or bell pepper leaf.",
            "internal_failure"
        )


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_disease(image_path):

    print("=" * 60, flush=True)

    print(
        "PREDICTION FUNCTION ENTERED",
        flush=True
    )

    try:

        # ====================================================
        # STEP 1 — VALIDATE LEAF
        # ====================================================

        is_leaf, validation_message = validate_leaf_image(
            image_path
        )

        if not is_leaf:

            failure = (
                validation_message
                if isinstance(validation_message, PredictionFailure)
                else PredictionFailure(validation_message, "invalid_leaf")
            )

            return (
                None,
                0,
                [],
                failure
            )

        # ====================================================
        # STEP 2 — OPEN IMAGE
        # ====================================================

        img = Image.open(
            image_path
        ).convert("RGB")

        print(
            "IMAGE OPENED SUCCESSFULLY",
            flush=True
        )

        # ====================================================
        # STEP 3 — RESIZE
        # ====================================================

        img = img.resize(
            (224, 224)
        )

        x = np.array(
            img
        ).astype(
            np.float32
        )

        x = np.expand_dims(
            x,
            axis=0
        )

        print(
            "IMAGE PREPARED",
            flush=True
        )

        # ====================================================
        # STEP 4 — MODEL PREDICTION
        # ====================================================

        print(
            "CALLING MODEL...",
            flush=True
        )

        pred = model(
            x,
            training=False
        )

        print(
            "MODEL CALL FINISHED",
            flush=True
        )

        pred = pred.numpy()

        # ====================================================
        # STEP 5 — PREDICTION
        # ====================================================

        predicted_index = int(
            np.argmax(
                pred[0]
            )
        )

        confidence = float(
            np.max(
                pred[0]
            ) * 100
        )

        disease = CLASS_NAMES[
            predicted_index
        ]

        # ====================================================
        # STEP 6 — TOP 3 PREDICTIONS
        # ====================================================

        top3_indices = (
            pred[0]
            .argsort()[-3:][::-1]
        )

        top3_predictions = []

        for idx in top3_indices:

            top3_predictions.append(
                {
                    "name": CLASS_NAMES[idx],
                    "confidence": round(
                        float(
                            pred[0][idx] * 100
                        ),
                        2
                    )
                }
            )

        # ====================================================
        # LOGGING
        # ====================================================

        print(
            f"DISEASE = {disease}",
            flush=True
        )

        print(
            f"CONFIDENCE = {confidence}",
            flush=True
        )

        print("=" * 60, flush=True)

        # ====================================================
        # SUCCESS
        # ====================================================

        return (
            disease,
            confidence,
            top3_predictions,
            None
        )

    except Exception as e:

        print(
            f"PREDICTION ERROR: {str(e)}",
            flush=True
        )

        validation_message = PredictionFailure(
            "Unable to process this image. "
            "Please upload a clear image of a tomato, "
            "potato, or bell pepper leaf.",
            "internal_failure"
        )

        return (
            None,
            0,
            [],
            validation_message
        )
