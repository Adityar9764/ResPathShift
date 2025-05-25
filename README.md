# ResNet-based Biopsy Classification and Domain Shift Detection Network

## 🔬 Project Summary

This project presents a comprehensive deep learning pipeline for **automated classification of biopsy images** using a **ResNet50-based Convolutional Neural Network (CNN)** and rigorous **domain shift analysis**. It leverages histopathological biopsy image data across lung and colon tissues to classify subtypes (e.g., adenocarcinoma, squamous cell carcinoma, normal) and examines domain shifts between training and external datasets using **statistical and geometric techniques** such as **MMD, PCA, t-SNE, Wasserstein Distance, KL Divergence, and Chi-Square tests**.

The project demonstrates **robust transfer learning**, **domain shift detection**, and **fine-tuning** for improved generalization. Extensive evaluations validate model performance and adaptability to unseen datasets, simulating real-world clinical conditions.

## 📁 Repository Structure

```
├── app.py
├── clean_dataset.py
├── dataset_setup.py
├── external_testing.py
├── external_testing_clahe.py
├── external_testing_CM.py
├── features-comparison-and-domain-shift-analysis_final.ipynb
├── Final_external_CM.py
├── fine_tune_clahe_model.py
├── load_model.py
├── misclassifiedImages.py
├── prepare_data.py
├── prior_shift_testing.ipynb
├── train_model.py
├── train_model_withoutshuffle.py
├── validation_testing.py
├── validation_testing_CM.py
├── visualizeImages.py
└── requirements.txt
```

## 🚀 Key Features

- **Biopsy Image Classification** using a pretrained ResNet50 CNN model.
- **Training with CLAHE-based preprocessing** for contrast normalization.
- **Prior and Covariate Shift Resolution** using folder restructuring and normalization.
- **Domain Shift Detection** through:
  - Maximum Mean Discrepancy (MMD)
  - Principal Component Analysis (PCA)
  - t-SNE Visualization
  - Wasserstein Distance
  - KL Divergence
  - Chi-Square Test
- **External Dataset Testing** to simulate real-world generalization.
- **Visualization of Misclassified Images** for model interpretation.
- **Confusion Matrix Generation** and performance metrics evaluation.
- **Web Integration Ready** (`app.py`) for future frontend deployment.

## 📦 How to Use This Project

### 1. Clone the Repository

```bash
git clone https://github.com/Adityar9764/ResPathShift.git
cd ResPathShift
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Dataset Setup

- Download the **training dataset** from this link: https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images
- Manually upload your **external testing dataset** to your environment or you can use the dataset used in our project : https://figshare.com/articles/dataset/LungHist700_A_Dataset_of_Histological_Images_for_Deep_Learning_in_Pulmonary_Pathology/25459174?file=45206104
- Datasets must follow the folder structure:
  ```
  ├── colon_image_sets/
  │   ├── colon_aca/
  │   └── colon_n/
  └── lung_image_sets/
      ├── lung_aca/
      ├── lung_n/
      └── lung_scc/
  ```

### 4. Train the Model (Optional)

To train the model from scratch with CLAHE preprocessing:

```bash
python train_model.py
```

### 5. Load and Test the Model

Download the trained model:
- [🔗 resnet50_biospy_model_clahe_v2.h5 – Download Link]
- [🔗 resnet50_biopsy_model_clahe_finetuned_final.h5 – Download Link]

Run inference and evaluate performance:

```bash
python external_testing.py
```

### 6. Perform Domain Shift Analysis

Use the provided notebook to explore domain shifts:

```bash
features-comparison-and-domain-shift-analysis_final.ipynb
```

## 📊 Example Evaluation Metrics

- Training Accuracy: ~93%
- Validation Testing Accuracy: ~88%
- External Testing Accuracy: ~39 ( which improved to around 62% after applying CLAHE and fine-tuning )
- Confusion Matrices and Misclassifications: Visualized and interpreted
- Domain Shift: Covariate shift identified and resolved through contrast normalization

## 🧠 Key Insights

- Domain adaptation significantly improves generalization.
- CLAHE enhances visual consistency between datasets.
- MMD and Wasserstein metrics quantify dataset drift effectively.
- Visualizations help interpret complex model behavior and failures.

## ✅ Acknowledgments

We sincerely thank the dataset creators for providing the **biopsy image datasets**, which were instrumental in the success of this project.

- 📦 **Training Dataset Source:** https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images
- 📦 **External Dataset Source:** https://figshare.com/articles/dataset/LungHist700_A_Dataset_of_Histological_Images_for_Deep_Learning_in_Pulmonary_Pathology/25459174?file=45206104

We do not claim ownership of the image datasets used. All rights and credits belong to the respective authors and institutions.


---

> 🔍 *This project represents a unique and comprehensive exploration of machine learning's intersection with pathology, tackling both model accuracy and real-world generalization through domain shift detection.*  
> _ResNet meets Real-World Biopsy Image Variability._
