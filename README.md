# AgroVision AI — Predictive Analytics

## Project Title
**Performance Evaluation and Comparative Analysis of Multiclass Classification Models for Plant Disease Detection**

## 1. Introduction

Plant diseases can affect agricultural productivity and crop quality. Machine learning and deep learning techniques can assist in identifying plant diseases by analyzing images of plant leaves.

AgroVision AI — Predictive Analytics focuses on evaluating the performance of multiclass image-classification models for identifying healthy and diseased plant leaves. The project uses the PlantVillage dataset and evaluates an existing EfficientNetB0-based classification model as the initial experimental baseline.

The Predictive Analytics project is focused on dataset analysis, model evaluation, performance metrics, and comparative analysis rather than simply developing a plant disease detection application.

## 2. Objectives

- Study a multiclass plant disease image-classification problem.
- Inspect the dataset, class labels, and class distribution.
- Verify the saved EfficientNetB0 model and its input/output dimensions.
- Evaluate the existing model using appropriate classification metrics.
- Analyze class-wise performance using precision, recall, F1-score, and a confusion matrix.
- Compare multiple classification models under consistent experimental conditions, if required by the assignment.
- Document the experimental methodology, results, limitations, and future improvements.

## 3. Dataset

The project uses the PlantVillage plant leaf image dataset obtained through Kaggle.

The dataset copy inspected for this project contains **20,638 valid images across 15 classes**, covering tomato, potato, and bell pepper leaves. The classes include healthy leaves and multiple disease categories.

The dataset was extracted and inspected in Google Colab. A separate 80:20 training/validation split was created for the Predictive Analytics evaluation using a random seed of 42.

- Total valid images identified: 20,638
- Number of classes: 15
- Training images in the reconstructed split: 16,504
- Validation images in the reconstructed split: 4,134
- Image input size: 224 × 224 pixels

**Dataset source:** [Insert the exact Kaggle dataset URL here.]

The exact dataset version and its relationship to the original PBL training data have not been independently established. The dataset is not included in this repository because of its size.

## 4. Machine Learning Approach

The primary task is **supervised multiclass image classification**.

The existing AgroVision AI system uses an EfficientNetB0-based deep learning model with a classification head consisting of global average pooling, a dense layer, dropout, and a 15-class softmax output.

The saved model was inspected and loaded successfully in Google Colab.

- Model architecture: EfficientNetB0-based sequential classifier
- Input shape: 224 × 224 × 3
- Output shape: 15 class scores
- Model file format: Keras HDF5 (`.h5`)

The existing model is used as the initial evaluation baseline. Additional classification models will be implemented and compared if required by the assignment and available computing resources.

## 5. Tools and Technologies

- Python
- Google Colab
- TensorFlow and Keras
- EfficientNetB0
- NumPy and Pandas
- Scikit-learn
- Matplotlib
- `split-folders`
- GitHub for version control and project collaboration

## 6. Experimental Methodology

The evaluation workflow consists of the following stages:

1. Extract and inspect the PlantVillage dataset.
2. Verify the number of classes, image counts, and class labels.
3. Reconstruct a separate 80:20 training/validation split using seed 42.
4. Configure image generators with a target size of 224 × 224 and a batch size of 32.
5. Apply the EfficientNet preprocessing function used by the original notebook.
6. Evaluate the saved EfficientNetB0 model on the reconstructed validation split.
7. Calculate overall and class-wise performance metrics.
8. Generate a confusion matrix and save the evaluation outputs.
9. Perform comparative analysis with an additional classification model if required.

The training generator includes image augmentation settings from the original notebook. Validation images are not augmented, and validation predictions are generated without shuffling so that predicted labels remain aligned with the true labels.

### Evaluation integrity

The original notebook used an 80:20 split with seed 42. The PA project recreated a split using the same ratio and seed in a separate directory.

However, the exact identity of the reconstructed split relative to the historical split has not been independently verified. The saved model's complete training history and exact relationship to the original evaluation artifact also remain unverified.

Consequently, the current evaluation is reported as **reconstructed validation performance**, not independent test-set performance.

## 7. Evaluation Metrics

The following metrics are used to assess the classification model:

- **Accuracy:** Proportion of all predictions that are correct.
- **Precision:** Proportion of predictions for a class that are correct.
- **Recall:** Proportion of actual examples of a class that are correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **Macro average:** Average metric across classes, giving every class equal weight.
- **Weighted average:** Average metric weighted by the number of examples in each class.
- **Confusion matrix:** Counts of correct and incorrect predictions for each actual and predicted class.

These metrics provide a more complete assessment than accuracy alone, particularly when class distributions and class-wise performance differ.

## 8. Initial Evaluation Results

The saved EfficientNetB0 model was evaluated on the reconstructed validation split containing 4,134 images across 15 classes.

| Metric | Result |
|---|---:|
| Validation accuracy | 75.91% |
| Macro precision | 0.84 |
| Macro recall | 0.70 |
| Macro F1-score | 0.68 |
| Weighted precision | 0.82 |
| Weighted recall | 0.76 |
| Weighted F1-score | 0.73 |

These values are taken from the actual evaluation output generated in Google Colab.

### Class-wise observations

The evaluation showed substantial variation in recall across classes.

| Class | Recall |
|---|---:|
| Pepper bell healthy | 1.00 |
| Tomato Yellow Leaf Curl Virus | 0.96 |
| Tomato Late Blight | 0.93 |
| Tomato Early Blight | 0.12 |
| Tomato Target Spot | 0.25 |
| Tomato Mosaic Virus | 0.04 |
| Tomato healthy | 0.30 |

The model performs strongly on some categories but misses many examples from others. This demonstrates why class-wise precision, recall, F1-score, and confusion-matrix analysis are important for multiclass classification.

The current results establish a baseline for further investigation. They do not, by themselves, establish why the errors occur or prove that one model is better than another.

### Evaluation artifacts

The generated outputs are available in the [`results/`](results/) directory:

- `classification_report.csv` — class-wise and aggregate classification metrics.
- `classification_report.txt` — readable classification report and validation accuracy.
- `confusion_matrix.csv` — confusion matrix counts.
- `confusion_matrix.png` — confusion matrix visualization.

## 9. Comparative Analysis — Pending

The initial evaluation covers the existing EfficientNetB0 model. A second classification model has not yet been evaluated as part of the documented PA experiment.

If comparative analysis is required, another suitable classifier will be selected and evaluated using consistent data separation and evaluation metrics.

Any comparison will report the actual experimental results. No comparative performance claims will be made until the additional model has been implemented and evaluated.

## 10. Limitations

- The current reported performance is based on a reconstructed validation split, not an independently verified test set.
- The exact relationship between the saved model, historical training process, and reconstructed split has not been fully established.
- Performance on PlantVillage images may not generalize to real-world field images with different lighting, backgrounds, camera conditions, and disease severity.
- Some classes have substantially lower recall than others.
- A comparative experiment with an additional classification model is still pending.
- No regression experiment or disease-severity prediction is included because reliable numerical ground-truth severity labels have not been established.

## 11. Repository Structure

```text
AgroVision-AI-PA/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── README.md
│   └── AgroVision_PA_Model_Verification (1).ipynb
└── results/
    ├── README.md
    ├── classification_report.csv
    ├── classification_report.txt
    ├── confusion_matrix.csv
    └── confusion_matrix.png
```

The repository contains the evaluation notebook and generated results, not the complete PlantVillage dataset or saved model artifact.

## 12. Reproducibility

The evaluation notebook was developed in Google Colab.

To reproduce the experiment, a collaborator will need:

1. Access to this GitHub repository.
2. The PlantVillage dataset used for the experiment.
3. The compatible saved EfficientNetB0 model artifact.
4. A Google Colab runtime with compatible Python and machine learning dependencies.
5. The evaluation notebook in `notebooks/`.

The dataset and model must be made available separately. The notebook should be checked for environment compatibility and its complete execution should be verified before claiming that the experiment is fully reproducible.

## 13. Future Scope

- Implement and evaluate an additional classification baseline.
- Compare models using accuracy, macro F1-score, weighted F1-score, and confusion matrices.
- Investigate classes with low recall and frequent misclassifications.
- Verify the original training split and model provenance.
- Evaluate on a genuinely independent test set if a suitable test set can be established.
- Investigate external field-image evaluation if an appropriately labelled dataset becomes available.

## 14. Project Status

**Current stage: Initial model evaluation completed; comparative analysis and reproducibility verification pending.**

The dataset was inspected, the saved model was loaded successfully, and the reconstructed validation evaluation was completed. The resulting classification report and confusion matrix have been uploaded to GitHub.

The next stages are to verify the experiment's reproducibility, implement the required model comparison, interpret the results, and prepare the final report and presentation.

## 15. Academic Integrity

All reported metrics are based on recorded experimental outputs. Validation results are not presented as independent test results, and unperformed experiments are explicitly identified as pending.

The project will distinguish verified results from assumptions, limitations, and future work.
