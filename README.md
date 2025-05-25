
```markdown
# ResNet-based Biopsy Classification and Domain Shift Detection Network

## 🔬 Project Summary

This project presents a comprehensive deep learning pipeline for **automated classification of biopsy images** using a **ResNet50-based Convolutional Neural Network (CNN)** and rigorous **domain shift analysis**. It leverages histopathological biopsy image data across lung and colon tissues to classify subtypes (e.g., adenocarcinoma, squamous cell carcinoma, normal) and examines domain shifts between training and external datasets using **statistical and geometric techniques** such as **MMD, PCA, t-SNE, Wasserstein Distance, KL Divergence, and Chi-Square tests**.

The project demonstrates **robust transfer learning**, **domain shift detection**, and **fine-tuning** for improved generalization. Extensive evaluations validate model performance and adaptability to unseen datasets, simulating real-world clinical conditions.

## 📁 Repository Structure

```

├── app.py
├── clean\_dataset.py
├── dataset\_setup.py
├── external\_testing.py
├── external\_testing\_clahe.py
├── external\_testing\_CM.py
├── features-comparison-and-domain-shift-analysis\_final.ipynb
├── Final\_external\_CM.py
├── fine\_tune\_clahe\_model.py
├── load\_model.py
├── misclassifiedImages.py
├── prepare\_data.py
├── prior\_shift\_testing.ipynb
├── train\_model.py
├── train\_model\_withoutshuffle.py
├── validation\_testing.py
├── validation\_testing\_CM.py
├── visualizeImages.py
└── requirements.txt

````

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
git clone https://github.com/yourusername/biopsy-domain-shift-detection.git
cd biopsy-domain-shift-detection
````

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Dataset Setup

* Download the **training dataset** from this link: \[🔗 Insert Training Dataset Link Here]
* Manually upload your **external testing dataset** to your environment: \[🔗 Insert External Dataset Link Here]
* Datasets must follow the folder structure:

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

* \[🔗 resnet50\_biospy\_model\_clahe\_v2.h5 – Download Link]
* \[🔗 resnet50\_biopsy\_model\_clahe\_finetuned\_final.h5 – Download Link]

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

* Training Accuracy: \~92%
* External Accuracy (after fine-tuning): Improved by 4–8%
* Confusion Matrices and Misclassifications: Visualized and interpreted
* Domain Shift: Covariate shift identified and resolved through contrast normalization

## 🧠 Key Insights

* Domain adaptation significantly improves generalization.
* CLAHE enhances visual consistency between datasets.
* MMD and Wasserstein metrics quantify dataset drift effectively.
* Visualizations help interpret complex model behavior and failures.

## ✅ Acknowledgments

We sincerely thank the dataset creators for providing the **biopsy image datasets**, which were instrumental in the success of this project.

* 📦 **Training Dataset Source:** \[🔗 Insert Link Here]
* 📦 **External Testing Dataset Source:** \[🔗 Insert Link Here]

We do not claim ownership of the image datasets used. All rights and credits belong to the respective authors and institutions.


> 🔍 *This project represents a unique and comprehensive exploration of machine learning's intersection with pathology, tackling both model accuracy and real-world generalization through domain shift detection.*
> *ResNet meets Real-World Biopsy Image Variability.*

```

