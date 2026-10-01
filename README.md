# 🛒 ShelfWatch
CAPSTONE-1 PROJECT

**ShelfWatch** is a food and grocery object-detection project built using **YOLOv8m-OBB**, **OpenCV**, **Supervision**, and **Google Colab**.

The project trains a custom YOLO model to detect and localize grocery/food products using **Oriented Bounding Boxes (OBB)**. It includes dataset preparation, model training, evaluation, class-wise performance analysis, image inference, and webcam-based detection.

---

## 🚀 Features

- 🛒 Food and grocery product detection
- 🎯 YOLOv8m Oriented Bounding Box (OBB) detection
- 📦 Custom dataset training
- 🔀 Automatic Train / Validation / Test splitting
- 📊 Dataset class-distribution analysis
- 🧠 Transfer learning using pretrained YOLOv8m-OBB
- 📈 Precision, Recall, F1, mAP50 and mAP50-95 evaluation
- 🔥 Confusion Matrix generation
- 📉 Precision-Recall curve visualization
- 🏷️ Per-class performance analysis
- 🖼️ Test-image detection
- 📤 User-uploaded image inference
- 📷 Google Colab webcam detection
- 💾 Google Drive integration for datasets and training results

---

## 🧠 Model

The project uses:

**YOLOv8m-OBB**

The model is initialized using:

```python
from ultralytics import YOLO

model = YOLO("yolov8m-obb.pt")
```

### Training Configuration

| Parameter | Value |
|---|---|
| Model | YOLOv8m-OBB |
| Task | Oriented Object Detection |
| Image Size | 640 |
| Batch Size | 32 |
| Maximum Epochs | 150 |
| Early Stopping Patience | 20 |
| Optimizer | Auto |
| Cosine LR Scheduling | Enabled |
| Random Seed | 42 |

---

## 📂 Dataset

The dataset contains grocery and food-product images with YOLO OBB annotations.

The notebook automatically extracts the dataset from **Google Drive**, validates its configuration, analyzes the available classes, and creates a clean dataset split.

### Dataset Split

| Dataset | Percentage |
|---|---:|
| Training | 80% |
| Validation | 10% |
| Testing | 10% |

Approximately:

```text
Training Images   : 30,984
Validation Images : 3,873
Testing Images    : 3,873
--------------------------------
Total Images      : 38,730
```

The final model configuration contains approximately **214 grocery/food classes**.

> The complete dataset is not uploaded to this repository because of its size.

---

## 📊 Model Performance

The final held-out test evaluation recorded in the notebook produced:

| Metric | Result |
|---|---:|
| **Precision** | **73.59%** |
| **Recall** | **63.74%** |
| **F1 Score** | **68.31%** |
| **mAP@50** | **70.76%** |
| **mAP@50-95** | **61.10%** |

### Understanding the Metrics

**Precision**

Measures how many detected objects were actually correct.

```text
Precision = TP / (TP + FP)
```

**Recall**

Measures how many actual objects in the images were successfully detected.

```text
Recall = TP / (TP + FN)
```

**F1 Score**

Balances Precision and Recall.

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**mAP@50**

Mean Average Precision at an IoU threshold of 0.50.

**mAP@50-95**

Mean Average Precision averaged across IoU thresholds from 0.50 to 0.95.

For object detection, **mAP is one of the primary model-evaluation metrics** rather than traditional classification accuracy.

---

## ⚙️ Project Workflow

```text
Dataset
   ↓
Dataset Extraction
   ↓
Class Verification
   ↓
Dataset Analysis
   ↓
80 / 10 / 10 Dataset Split
   ↓
YOLOv8m-OBB
   ↓
Model Training
   ↓
Best Model Selection
   ↓
Validation & Testing
   ↓
Performance Analysis
   ↓
Image / Webcam Detection
```

The notebook performs the following steps:

1. Install required libraries
2. Mount Google Drive
3. Extract the dataset
4. Load `data.yaml`
5. Verify class names
6. Analyze class distribution
7. Create clean Train / Validation / Test splits
8. Load pretrained YOLOv8m-OBB
9. Train the custom model
10. Locate the best trained model
11. Evaluate the model
12. Generate confusion matrices
13. Generate Precision-Recall curves
14. Calculate per-class performance
15. Run detection on test images
16. Run detection on uploaded images
17. Perform webcam inference in Google Colab

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| YOLOv8 | Object detection |
| Ultralytics | YOLO training and inference |
| YOLO OBB | Oriented bounding-box detection |
| OpenCV | Image processing |
| Supervision | Detection utilities |
| NumPy | Numerical operations |
| Pandas | Metrics/data processing |
| Matplotlib | Graphs and visualization |
| PyYAML | Dataset configuration |
| Google Colab | Training environment |
| Google Drive | Dataset and model storage |

---

## 📁 Repository Structure

```text
ShelfWatch/
│
├── shelfwatch.ipynb
│
├── README.md
│
└── requirements.txt
```

Large datasets and generated training files are intentionally excluded from the repository.

Examples:

```text
dataset/
dataset_clean/
runs/
weights/
*.pt
```

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/jyothiradithya2005/ShelfWatch.git
```

Enter the project directory:

```bash
cd ShelfWatch
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Or install the main libraries directly:

```bash
pip install ultralytics supervision
```

---

## ▶️ Running the Project

The recommended environment is **Google Colab**.

### Step 1

Open:

```text
shelfwatch.ipynb
```

in Google Colab.

### Step 2

Place the dataset ZIP file in your Google Drive.

The notebook currently expects a path similar to:

```text
/content/drive/MyDrive/Complete Food.v1-yolov8-obb.zip
```

Change the path in the notebook if your dataset is stored somewhere else.

### Step 3

Run the notebook cells sequentially.

```text
Runtime → Run all
```

### Step 4

The notebook will:

```text
Extract Dataset
      ↓
Prepare Dataset
      ↓
Train YOLO
      ↓
Evaluate Model
      ↓
Generate Metrics
      ↓
Run Detection
```

---

## 🔍 Image Detection

The trained model can be used to detect grocery products in new images.

Example:

```python
from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict(
    source="test_image.jpg",
    conf=0.25
)
```

---

## 📷 Webcam Detection

ShelfWatch also contains utilities for using a webcam directly inside **Google Colab**.

This allows the trained model to process images captured using the browser camera.

---

## 📈 Model Evaluation

The notebook generates several evaluation outputs, including:

### Confusion Matrix

Used to analyze which classes are detected correctly and which classes are commonly confused.

### Precision-Recall Curve

Shows the relationship between:

```text
Precision ↔ Recall
```

at different confidence thresholds.

### Class-wise Metrics

ShelfWatch also calculates performance separately for individual grocery classes.

This helps identify:

- Strong-performing classes
- Weak-performing classes
- Classes with low recall
- Classes with low precision
- Classes requiring additional training data

---

## 🎯 Applications

ShelfWatch can be extended for applications such as:

- 🛒 Smart supermarket shelves
- 📦 Inventory monitoring
- 📉 Low-stock detection
- 🏪 Retail shelf monitoring
- 🤖 Automated product recognition
- 📊 Shelf analytics
- 🧾 Smart checkout systems
- 📷 Real-time grocery detection

---

## 🔮 Future Improvements

Future versions of ShelfWatch can include:

- Real-time inventory counting
- Automatic low-stock alerts
- Product quantity tracking
- Duplicate-detection prevention
- Improved detection of small/far-away products
- Larger and more balanced training datasets
- Removal or retraining of weak classes
- Web dashboard for inventory monitoring
- Real-time camera streaming
- Database integration
- Notification system for low-stock products

---

## ⚠️ Notes

- Dataset files are not included because of their large size.
- Trained `.pt` weights may be stored separately.
- Google Drive is used to store datasets and training outputs.
- Model results can vary between training runs.
- Performance depends heavily on dataset quality, class balance, image quality, and annotations.

---

## 👨‍💻 Author

**Jyothiradithya Sagiraju**

GitHub: `jyothiradithya2005`

---

## ⭐ Support

If you find **ShelfWatch** useful, consider giving the repository a ⭐.

**ShelfWatch — AI-powered grocery and shelf monitoring using YOLO.**
