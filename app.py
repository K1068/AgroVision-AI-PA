from flask import Flask, jsonify, render_template, request, send_file, Response
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge

from PIL import Image, UnidentifiedImageError

import csv
from functools import lru_cache
from io import BytesIO
import os
import time
import sqlite3
import uuid
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo
import psutil

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import getSampleStyleSheet

from predict import predict_disease
from disease_info import DISEASE_INFO
from recommendation_engine import RECOMMENDATIONS


app = Flask(__name__)
INDIA_TIMEZONE = ZoneInfo("Asia/Kolkata")

UPLOAD_FOLDER = "static/uploads"
MAX_UPLOAD_SIZE = 8 * 1024 * 1024
MAX_IMAGE_SIDE = 4096
MAX_IMAGE_PIXELS = 16_000_000
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_SIZE

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

LAST_REPORT = {}


class InvalidUploadedImage(ValueError):

    def __init__(self, message, category="invalid_image", status_code=400):
        super().__init__(message)
        self.category = category
        self.status_code = status_code


def validate_uploaded_image(image):

    image_data = image.stream.read(MAX_UPLOAD_SIZE + 1)

    if len(image_data) > MAX_UPLOAD_SIZE:
        raise InvalidUploadedImage(
            "Image exceeds the 8 MiB upload limit.",
            category="upload_too_large",
            status_code=413
        )

    try:

        with warnings.catch_warnings():
            warnings.simplefilter(
                "error",
                Image.DecompressionBombWarning
            )

            with Image.open(BytesIO(image_data)) as uploaded_image:

                image_format = uploaded_image.format

                if image_format not in {"JPEG", "PNG"}:
                    raise InvalidUploadedImage(
                        "Please upload a valid JPEG or PNG image.",
                        category="unsupported_format",
                        status_code=415
                    )

                width, height = uploaded_image.size

                if width < 100 or height < 100:
                    raise InvalidUploadedImage(
                        "Image dimensions must be at least 100×100 pixels."
                    )

                if width > MAX_IMAGE_SIDE or height > MAX_IMAGE_SIDE:
                    raise InvalidUploadedImage(
                        "Image dimensions must not exceed 4096 pixels per side."
                    )

                if width * height > MAX_IMAGE_PIXELS:
                    raise InvalidUploadedImage(
                        "Image dimensions must not exceed 16 megapixels."
                    )

                uploaded_image.verify()

            with Image.open(BytesIO(image_data)) as decoded_image:
                decoded_image.load()

    except InvalidUploadedImage:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning,
            UnidentifiedImageError, OSError) as error:
        raise InvalidUploadedImage(
            "Please upload a valid JPEG or PNG image."
        ) from error
    except Exception as error:
        raise InvalidUploadedImage(
            "Please upload a valid JPEG or PNG image.",
            category="internal_failure",
            status_code=500
        ) from error

    extension = ".jpg" if image_format == "JPEG" else ".png"
    return image_data, extension


def remove_upload(filepath):

    if not filepath:
        return

    try:
        os.remove(filepath)
    except FileNotFoundError:
        pass
    except OSError:
        app.logger.exception("Unable to remove rejected uploaded image")


# ==================================================
# DATABASE
# ==================================================

def init_db():

    print("DATABASE INITIALIZED", flush=True)

    conn = sqlite3.connect("agrovision.db")

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            disease TEXT,
            confidence REAL,
            image_path TEXT,
            analysis_time TEXT
        )
    """)

    conn.commit()
    conn.close()


init_db()


# ==================================================
# HOME PAGE
# ==================================================

@app.route("/")
def home():

    print("HOME PAGE OPENED", flush=True)

    return render_template(
        "index.html"
    )


# ==================================================
# MEMORY CHECK
# ==================================================

@app.route("/memory")
def memory():

    process = psutil.Process()

    return {
        "memory_mb": round(
            process.memory_info().rss / 1024 / 1024,
            2
        )
    }


# ==================================================
# IMAGE UPLOAD
# ==================================================

@app.route("/upload", methods=["POST"])
def upload():

    global LAST_REPORT

    print("=" * 60, flush=True)
    print("UPLOAD REQUEST RECEIVED", flush=True)

    start_time = time.time()
    filepath = None
    history_committed = False

    try:

        # ------------------------------------------
        # GET IMAGE
        # ------------------------------------------

        image = request.files.get("leaf_image")

        if not image:

            return "Please select an image to upload.", 400


        print(
            "IMAGE RECEIVED",
            flush=True
        )

        try:
            image_data, extension = validate_uploaded_image(image)
        except InvalidUploadedImage as error:
            return str(error), 400


        # ------------------------------------------
        # SAVE IMAGE
        # ------------------------------------------

        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

        while True:

            generated_filename = f"{uuid.uuid4().hex}{extension}"
            candidate_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                generated_filename
            )

            try:
                with open(candidate_path, "xb") as saved_image:
                    filepath = candidate_path
                    saved_image.write(image_data)
                break
            except FileExistsError:
                continue

        print(
            f"IMAGE SAVED: {filepath}",
            flush=True
        )


        # ------------------------------------------
        # PREDICTION
        # ------------------------------------------

        print(
            "CALLING PREDICT_DISEASE()",
            flush=True
        )

        disease, confidence, top3_predictions, validation_error = predict_disease(
            filepath
        )


        # ------------------------------------------
        # INVALID IMAGE
        # ------------------------------------------

        if validation_error:

            print(
                f"VALIDATION ERROR: {validation_error}",
                flush=True
            )

            remove_upload(filepath)
            filepath = None

            return render_template(
                "result.html",
                validation_error=validation_error
            )


        print(
            "PREDICTION RETURNED",
            flush=True
        )


        # ------------------------------------------
        # DISEASE INFORMATION
        # ------------------------------------------

        info = DISEASE_INFO.get(
            disease,
            {
                "symptoms": "Information not available",
                "treatment": "Information not available",
                "prevention": "Information not available"
            }
        )


        # ------------------------------------------
        # RECOMMENDATION
        # ------------------------------------------

        recommendation = RECOMMENDATIONS.get(
            disease,
            {
                "english": "No recommendation available.",
                "hindi": "कोई सुझाव उपलब्ध नहीं है।"
            }
        )


        # ------------------------------------------
        # ROUND CONFIDENCE
        # ------------------------------------------

        confidence = round(
            confidence,
            2
        )


        # ------------------------------------------
        # ANALYSIS TIME
        # ------------------------------------------

        analysis_time = datetime.now(
            INDIA_TIMEZONE
        ).strftime(
            "%d-%m-%Y %I:%M %p"
        )


        # ------------------------------------------
        # SAVE TO DATABASE
        # ------------------------------------------

        conn = sqlite3.connect("agrovision.db")

        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO history
                (
                    disease,
                    confidence,
                    image_path,
                    analysis_time
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    disease,
                    confidence,
                    filepath,
                    analysis_time
                )
            )

            conn.commit()
            history_committed = True
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        LAST_REPORT = {
            "disease": disease,
            "confidence": confidence,
            "symptoms": info["symptoms"],
            "treatment": info["treatment"],
            "prevention": info["prevention"],
            "analysis_time": analysis_time
        }


        # ------------------------------------------
        # TOTAL REQUEST TIME
        # ------------------------------------------

        print(
            f"TOTAL REQUEST TIME: "
            f"{round(time.time() - start_time, 2)} sec",
            flush=True
        )


        # ------------------------------------------
        # RESULT PAGE
        # ------------------------------------------

        return render_template(

            "result.html",

            disease=disease,

            confidence=confidence,

            analysis_time=analysis_time,

            image_path=filepath,

            symptoms=info["symptoms"],

            treatment=info["treatment"],

            prevention=info["prevention"],

            top3_predictions=top3_predictions,

            recommendation_english=recommendation["english"],

            recommendation_hindi=recommendation["hindi"],

            validation_error=None

        )


    except RequestEntityTooLarge:
        return "Upload is too large. The maximum request size is 8 MiB.", 413

    except Exception:

        print(
            "ERROR IN APP.PY",
            flush=True
        )

        app.logger.exception("Unable to process uploaded image")

        if not history_committed:
            remove_upload(filepath)

        return "Unable to process the uploaded image. Please try again.", 500


def api_error_response(status_code, code, message):

    return jsonify(
        success=False,
        error={
            "code": code,
            "message": message
        }
    ), status_code


@app.route("/api/predict", methods=["POST"])
def api_predict():

    filepath = None
    history_committed = False

    try:

        image = request.files.get("leaf_image")

        if image is None or not image.filename:
            return api_error_response(
                400,
                "missing_file",
                "Provide an image in the 'leaf_image' field."
            )

        image_data, extension = validate_uploaded_image(image)

        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

        while True:
            generated_filename = f"{uuid.uuid4().hex}{extension}"
            candidate_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                generated_filename
            )

            try:
                with open(candidate_path, "xb") as saved_image:
                    filepath = candidate_path
                    saved_image.write(image_data)
                break
            except FileExistsError:
                continue

        disease, confidence, top3_predictions, validation_error = (
            predict_disease(filepath)
        )

        if validation_error:
            remove_upload(filepath)
            filepath = None

            category = getattr(
                validation_error,
                "category",
                "internal_failure"
            )

            if category == "invalid_leaf":
                return api_error_response(
                    422,
                    "invalid_leaf",
                    "The image does not appear to be a supported plant leaf."
                )
            if category == "invalid_image":
                return api_error_response(
                    400,
                    "invalid_image",
                    "The image could not be processed. Upload a valid JPEG or PNG."
                )

            app.logger.error(
                "Prediction failed with category %s",
                category
            )
            return api_error_response(
                500,
                "internal_error",
                "Unable to process the image. Please try again."
            )

        info = DISEASE_INFO.get(
            disease,
            {
                "symptoms": "Information not available",
                "treatment": "Information not available",
                "prevention": "Information not available"
            }
        )
        recommendation = RECOMMENDATIONS.get(
            disease,
            {
                "english": "No recommendation available.",
                "hindi": "कोई सुझाव उपलब्ध नहीं है।"
            }
        )
        confidence = round(confidence, 2)
        analysis_time = datetime.now(
            INDIA_TIMEZONE
        ).strftime(
            "%d-%m-%Y %I:%M %p"
        )

        top_3 = [
            {
                "predicted_class": item["name"],
                "model_score_percent": item["confidence"]
            }
            for item in top3_predictions
        ]
        response = jsonify(
            success=True,
            prediction={
                "predicted_class": disease,
                "model_score_percent": confidence,
                "top_3": top_3
            },
            disease_info={
                "symptoms": info["symptoms"],
                "treatment": info["treatment"],
                "prevention": info["prevention"]
            },
            recommendations={
                "english": recommendation["english"],
                "hindi": recommendation["hindi"]
            },
            analysis_time=analysis_time,
            score_note=(
                "This score reflects the model's output. It is not a "
                "confirmed diagnosis or a measure of disease severity, "
                "and predictions can be incorrect."
            )
        )

        conn = sqlite3.connect("agrovision.db")

        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO history
                (
                    disease,
                    confidence,
                    image_path,
                    analysis_time
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    disease,
                    confidence,
                    filepath,
                    analysis_time
                )
            )
            conn.commit()
            history_committed = True
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

        return response

    except RequestEntityTooLarge:
        return api_error_response(
            413,
            "upload_too_large",
            "Upload is too large. The maximum request size is 8 MiB."
        )
    except InvalidUploadedImage as error:
        if error.category == "internal_failure":
            app.logger.exception("Unable to validate API image")
            return api_error_response(
                500,
                "internal_error",
                "Unable to process the image. Please try again."
            )
        return api_error_response(
            error.status_code,
            error.category,
            str(error)
        )
    except BadRequest:
        return api_error_response(
            400,
            "invalid_request",
            "Request must contain a multipart image upload."
        )
    except Exception:
        app.logger.exception("Unable to process API image prediction")

        if not history_committed:
            remove_upload(filepath)

        return api_error_response(
            500,
            "internal_error",
            "Unable to process the image. Please try again."
        )


# ==================================================
# PDF REPORT
# ==================================================

@app.route("/download-report")
def download_report():

    pdf_file = "AgroVision_Report.pdf"

    doc = SimpleDocTemplate(
        pdf_file
    )

    styles = getSampleStyleSheet()

    content = [

        Paragraph(
            "AgroVision AI Disease Report",
            styles["Title"]
        ),

        Spacer(1, 20),

        Paragraph(
            f"Predicted class: "
            f"{LAST_REPORT.get('disease', 'N/A')}",
            styles["Normal"]
        ),

        Paragraph(
            f"Model confidence score: "
            f"{LAST_REPORT.get('confidence', 'N/A')}%",
            styles["Normal"]
        ),

        Paragraph(
            f"Date: "
            f"{LAST_REPORT.get('analysis_time', 'N/A')}",
            styles["Normal"]
        ),

        Spacer(1, 15),

        Paragraph(
            "Symptoms",
            styles["Heading2"]
        ),

        Paragraph(
            LAST_REPORT.get(
                "symptoms",
                "N/A"
            ),
            styles["Normal"]
        ),

        Spacer(1, 10),

        Paragraph(
            "Treatment",
            styles["Heading2"]
        ),

        Paragraph(
            LAST_REPORT.get(
                "treatment",
                "N/A"
            ),
            styles["Normal"]
        ),

        Spacer(1, 10),

        Paragraph(
            "Prevention",
            styles["Heading2"]
        ),

        Paragraph(
            LAST_REPORT.get(
                "prevention",
                "N/A"
            ),
            styles["Normal"]
        )

    ]


    doc.build(
        content
    )


    return send_file(
        pdf_file,
        as_attachment=True
    )


# ==================================================
# HISTORY PAGE
# ==================================================

@app.route("/history")
def history():

    conn = sqlite3.connect(
        "agrovision.db"
    )

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            disease,
            confidence,
            image_path,
            analysis_time
        FROM history
        ORDER BY id DESC
        """
    )

    records = cursor.fetchall()

    conn.close()


    return render_template(
        "history.html",
        records=records
    )


# ==================================================
# DOWNLOAD CSV
# ==================================================

@app.route("/download-csv")
def download_csv():

    conn = sqlite3.connect(
        "agrovision.db"
    )

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            disease,
            confidence,
            image_path,
            analysis_time
        FROM history
        ORDER BY id DESC
        """
    )

    rows = cursor.fetchall()

    conn.close()


    def generate():

        yield (
            "Predicted class,"
            "Model confidence score,"
            "Image Path,"
            "Analysis Time\n"
        )


        for row in rows:

            yield (
                f"{row[0]},"
                f"{row[1]},"
                f"{row[2]},"
                f"{row[3]}\n"
            )


    return Response(

        generate(),

        mimetype="text/csv",

        headers={
            "Content-Disposition":
            "attachment; "
            "filename=agrovision_history.csv"
        }

    )


# ==================================================
# DASHBOARD
# ==================================================

VALIDATION_SPLIT_IMAGES = 4134


def calculate_confusion_metrics(class_names, counts_by_row):

    class_count = len(class_names)

    if class_count == 0 or len(counts_by_row) != class_count:
        raise ValueError(
            "The confusion matrix must have one row and column per class."
        )

    if any(len(row) != class_count for row in counts_by_row):
        raise ValueError(
            "The confusion matrix must be square."
        )

    if any(
        not isinstance(count, int) or count < 0
        for row in counts_by_row
        for count in row
    ):
        raise ValueError(
            "The confusion matrix must contain nonnegative integer counts."
        )

    total_count = sum(
        sum(row_counts)
        for row_counts in counts_by_row
    )

    column_totals = [
        sum(counts_by_row[row_index][column_index]
            for row_index in range(class_count))
        for column_index in range(class_count)
    ]

    def divide_or_zero(numerator, denominator):
        return numerator / denominator if denominator else 0.0

    per_class_metrics = []

    for class_index, class_name in enumerate(class_names):

        true_positive = counts_by_row[class_index][class_index]
        false_negative = sum(counts_by_row[class_index]) - true_positive
        false_positive = column_totals[class_index] - true_positive
        true_negative = (
            total_count
            - true_positive
            - false_positive
            - false_negative
        )

        precision = divide_or_zero(
            true_positive,
            true_positive + false_positive
        )
        recall = divide_or_zero(
            true_positive,
            true_positive + false_negative
        )

        per_class_metrics.append(
            {
                "class_name": class_name,
                "support": sum(counts_by_row[class_index]),
                "tp": true_positive,
                "tn": true_negative,
                "fp": false_positive,
                "fn": false_negative,
                "accuracy": divide_or_zero(
                    true_positive + true_negative,
                    total_count
                ),
                "precision": precision,
                "recall": recall,
                "f1_score": divide_or_zero(
                    2 * precision * recall,
                    precision + recall
                ),
                "specificity": divide_or_zero(
                    true_negative,
                    true_negative + false_positive
                ),
                "error_rate": divide_or_zero(
                    false_positive + false_negative,
                    total_count
                ),
                "false_positive_rate": divide_or_zero(
                    false_positive,
                    false_positive + true_negative
                ),
                "false_negative_rate": divide_or_zero(
                    false_negative,
                    false_negative + true_positive
                )
            }
        )

    macro_metric_names = (
        "precision",
        "recall",
        "f1_score",
        "specificity",
        "false_positive_rate",
        "false_negative_rate"
    )

    macro_metrics = {
        metric_name: sum(
            class_metrics[metric_name]
            for class_metrics in per_class_metrics
        ) / class_count
        for metric_name in macro_metric_names
    }

    correct_predictions = sum(
        counts_by_row[index][index]
        for index in range(class_count)
    )

    return {
        "total_count": total_count,
        "accuracy": divide_or_zero(correct_predictions, total_count),
        "error_rate": divide_or_zero(
            total_count - correct_predictions,
            total_count
        ),
        "macro_metrics": macro_metrics,
        "per_class_metrics": per_class_metrics
    }


@lru_cache(maxsize=1)
def load_confusion_matrix():

    matrix_path = os.path.join(
        app.root_path,
        "results",
        "confusion_matrix.csv"
    )

    with open(
        matrix_path,
        newline="",
        encoding="utf-8"
    ) as matrix_file:

        reader = csv.reader(matrix_file)
        header = next(reader, None)

        if not header or header[0] != "":
            raise ValueError(
                "The confusion matrix CSV must have an empty first header "
                "cell followed by the predicted class names."
            )

        class_names = header[1:]

        if (
            len(class_names) != 15
            or any(not name for name in class_names)
            or len(set(class_names)) != len(class_names)
        ):
            raise ValueError(
                "The confusion matrix CSV must contain 15 unique class names."
            )

        csv_rows = list(reader)

    if len(csv_rows) != len(class_names):
        raise ValueError(
            "The confusion matrix CSV must contain one row per class."
        )

    counts_by_row = []

    for row_index, csv_row in enumerate(csv_rows):

        if (
            len(csv_row) != len(class_names) + 1
            or csv_row[0] != class_names[row_index]
        ):
            raise ValueError(
                "The confusion matrix CSV row labels and counts must match "
                "the class headers and their order."
            )

        try:
            counts = [int(value) for value in csv_row[1:]]
        except ValueError as error:
            raise ValueError(
                "The confusion matrix CSV must contain whole-number counts."
            ) from error

        if any(count < 0 for count in counts):
            raise ValueError(
                "The confusion matrix CSV cannot contain negative counts."
            )

        counts_by_row.append(counts)

    total_count = sum(
        sum(row_counts)
        for row_counts in counts_by_row
    )

    if total_count != VALIDATION_SPLIT_IMAGES:
        raise ValueError(
            "The confusion matrix counts must total "
            f"{VALIDATION_SPLIT_IMAGES} validation images; found {total_count}."
        )

    maximum_count = max(
        max(row_counts)
        for row_counts in counts_by_row
    )

    matrix_rows = []

    for row_index, row_counts in enumerate(counts_by_row):

        cells = []

        for column_index, count in enumerate(row_counts):

            intensity = 0.08 + 0.92 * count / maximum_count

            cells.append(
                {
                    "predicted_class": class_names[column_index],
                    "count": count,
                    "intensity": f"{intensity:.3f}",
                    "text_color": (
                        "#ffffff"
                        if intensity >= 0.55
                        else "#1b5e20"
                    ),
                    "is_diagonal": row_index == column_index
                }
            )

        matrix_rows.append(
            {
                "actual_class": class_names[row_index],
                "cells": cells
            }
        )

    return class_names, matrix_rows, maximum_count, total_count


@app.route("/dashboard")
def dashboard():

    conn = sqlite3.connect(
        "agrovision.db"
    )

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            disease,
            confidence
        FROM history
        """
    )

    records = cursor.fetchall()

    conn.close()


    total_analyses = len(
        records
    )

    healthy_count = 0

    diseased_count = 0

    total_model_score = 0

    disease_stats = {}


    for disease, confidence in records:

        total_model_score += confidence


        # ------------------------------------------
        # Disease count
        # ------------------------------------------

        if disease in disease_stats:

            disease_stats[disease] += 1

        else:

            disease_stats[disease] = 1


        # ------------------------------------------
        # Healthy vs Diseased
        # ------------------------------------------

        if "healthy" in disease.lower():

            healthy_count += 1

        else:

            diseased_count += 1


    # ------------------------------------------
    # Average model score
    # ------------------------------------------

    if total_analyses > 0:

        average_model_score = round(

            total_model_score /
            total_analyses,

            2

        )

    else:

        average_model_score = 0


    # ------------------------------------------
    # Chart data
    # ------------------------------------------

    chart_labels = list(
        disease_stats.keys()
    )

    chart_values = list(
        disease_stats.values()
    )

    evaluation_available = True
    evaluation_metrics = []
    per_class_metrics = []
    validation_split_images = 0
    confusion_class_names = []
    confusion_matrix_rows = []
    confusion_matrix_max = 0
    confusion_matrix_total = 0

    try:
        (
            confusion_class_names,
            confusion_matrix_rows,
            confusion_matrix_max,
            confusion_matrix_total
        ) = load_confusion_matrix()
    except FileNotFoundError:
        evaluation_available = False
        app.logger.warning(
            "Offline evaluation unavailable: results/confusion_matrix.csv "
            "was not found."
        )
    else:
        counts_by_row = [
            [cell["count"] for cell in matrix_row["cells"]]
            for matrix_row in confusion_matrix_rows
        ]
        metrics = calculate_confusion_metrics(
            confusion_class_names,
            counts_by_row
        )
        validation_split_images = metrics["total_count"]
        macro_metrics = metrics["macro_metrics"]
        per_class_metrics = metrics["per_class_metrics"]
        evaluation_metrics = [
            {
                "label": "Accuracy",
                "value": "{:.2%}".format(metrics["accuracy"])
            },
            {
                "label": "Macro precision (OvR)",
                "value": "{:.2%}".format(macro_metrics["precision"])
            },
            {
                "label": "Macro recall (OvR)",
                "value": "{:.2%}".format(macro_metrics["recall"])
            },
            {
                "label": "Macro F1-score (OvR)",
                "value": "{:.2%}".format(macro_metrics["f1_score"])
            },
            {
                "label": "Macro specificity (OvR)",
                "value": "{:.2%}".format(macro_metrics["specificity"])
            },
            {
                "label": "Error rate (overall)",
                "value": "{:.2%}".format(metrics["error_rate"])
            },
            {
                "label": "Macro FPR (OvR)",
                "value": "{:.2%}".format(
                    macro_metrics["false_positive_rate"]
                )
            },
            {
                "label": "Macro FNR (OvR)",
                "value": "{:.2%}".format(
                    macro_metrics["false_negative_rate"]
                )
            }
        ]

    return render_template(

        "dashboard.html",

        total_analyses=total_analyses,

        healthy_count=healthy_count,

        diseased_count=diseased_count,

        average_model_score=average_model_score,

        chart_labels=chart_labels,

        chart_values=chart_values,

        evaluation_available=evaluation_available,

        evaluation_metrics=evaluation_metrics,

        validation_split_images=validation_split_images,

        confusion_class_names=confusion_class_names,

        per_class_metrics=per_class_metrics,

        confusion_matrix_rows=confusion_matrix_rows,

        confusion_matrix_max=confusion_matrix_max,

        confusion_matrix_total=confusion_matrix_total

    )


# ==================================================
# TEST
# ==================================================

@app.route("/test")
def test():

    return "Flask is working"


# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=10000

    )
