# 🛒 ShelfWatch

**ShelfWatch** is an AI-powered grocery and shelf-monitoring application that detects grocery items from images, videos, and live camera streams, counts detected objects, tracks stock levels, and can send SMS alerts when items run low.

The project combines a **custom-trained grocery detection model** with a **general YOLO model** to provide grocery detection, everyday-object detection, and people detection.

---

## 🚀 Features

- 🛒 Grocery and food-product detection
- 📦 Grocery item counting
- 📷 Photo/image detection
- 🎥 Video detection
- 📹 Live webcam detection
- 🌐 RTSP/HTTP camera stream support
- 👤 People detection
- 🔍 Everyday-object detection
- 📊 Live stock counting
- 📈 Cumulative object counting
- ⚠️ Low-stock detection
- 📱 SMS alerts using Twilio
- 🔗 Reorder links in SMS alerts
- 📋 Stock and alert history
- 🎯 Custom grocery detection model
- 🤖 YOLO COCO model for general objects
- 📊 Model evaluation and performance analysis
- 🧠 YOLOv8m-OBB training pipeline
- ☁️ Google Colab and Google Drive support

---

# 🧠 AI Models

ShelfWatch combines two detection models.

## 1. Custom Grocery Model

```text
best.pt
```

This is the custom-trained grocery detection model used to identify food and grocery products.

The training pipeline uses:

```text
YOLOv8m-OBB
```

The model was trained on a custom grocery/food dataset containing approximately **214 classes**.

The dataset contains approximately:

```text
Training Images   : 30,984
Validation Images : 3,873
Testing Images    : 3,873
Total Images      : 38,730
```

The dataset was divided into:

```text
80% Training
10% Validation
10% Testing
```

---

## 2. General YOLO Model

```text
yolo11n.pt
```

The general YOLO model is used for everyday objects and people.

It can detect objects such as:

```text
person
bottle
cup
bowl
phone
backpack
chair
laptop
and other COCO objects
```

The model is downloaded automatically by Ultralytics if it is not already available.

---

# 🔄 Detection Architecture

ShelfWatch combines both models:

```text
                    ┌─────────────────────┐
                    │      Input          │
                    │ Photo / Video /     │
                    │ Live Camera / RTSP  │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
        ┌─────────────────┐       ┌─────────────────┐
        │   best.pt       │       │   yolo11n.pt    │
        │ Custom Grocery  │       │ COCO Model      │
        │ Detection       │       │ Objects/People  │
        └────────┬────────┘       └────────┬────────┘
                 │                         │
                 └────────────┬────────────┘
                              ▼
                    ┌───────────────────┐
                    │ Detection +       │
                    │ Object Tracking   │
                    └─────────┬─────────┘
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼
          Live Stock Count          Cumulative Count
                 │                         │
                 └────────────┬────────────┘
                              ▼
                     ┌─────────────────┐
                     │ Low Stock Check │
                     └────────┬────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │ Twilio SMS Alert│
                     └─────────────────┘
```

---

# 📊 Model Training

The custom grocery model was trained using **YOLOv8m-OBB**.

### Training configuration

| Parameter | Value |
|---|---:|
| Model | YOLOv8m-OBB |
| Task | Oriented Object Detection |
| Image Size | 640 |
| Batch Size | 32 |
| Maximum Epochs | 150 |
| Early Stopping Patience | 20 |
| Optimizer | Auto |
| Cosine LR Scheduling | Enabled |
| Random Seed | 42 |
| Dataset Classes | 214 |

---

# 📈 Model Evaluation

The final held-out test evaluation recorded in the training notebook produced:

| Metric | Result |
|---|---:|
| **mAP@50** | **70.76%** |
| **mAP@50-95** | **61.10%** |
| **Precision** | **73.59%** |
| **Recall** | **63.74%** |
| **F1 Score** | **68.31%** |

The training notebook also includes:

- Confusion matrix
- Precision-Recall curves
- Class-wise metrics
- Precision analysis
- Recall analysis
- mAP analysis
- Test-image inference
- Webcam inference

---

# 🖥️ Application

The ShelfWatch application is built using **Streamlit**.

Run it using:

```bash
streamlit run app.py
```

The application opens at:

```text
http://localhost:8501
```

---

# 📋 Application Modes

## Objects Mode

Objects mode detects:

- Grocery items
- Food products
- Everyday COCO objects

People can be excluded from the grocery stock count.

---

## People Mode

People mode uses the general YOLO model to detect and count people.

This can be used for:

- People counting
- Store occupancy monitoring
- Customer traffic analysis

---

# 📷 Photo Detection

The **Photo** tab allows you to:

- Upload an image
- Take an image using a camera
- Detect grocery products
- Count detected objects
- Display bounding boxes
- Display stock levels

---

# 🎥 Video Detection

The **Video** tab supports:

```text
MP4
MOV
AVI
MKV
```

Upload a video and select:

```text
Analyse video
```

The application processes the video and can produce a downloadable version with detection boxes.

You can also configure:

```text
Analyse every Nth frame
```

to trade processing speed for detection frequency.

---

# 📹 Live Camera Detection

The **Live Camera** tab supports:

### Computer webcam

Enter:

```text
0
```

for the default webcam.

### Network camera

You can also provide an:

```text
RTSP
```

or

```text
HTTP
```

camera stream URL.

Then enable:

```text
Start live detection
```

---

# 📊 Live vs Cumulative Counting

ShelfWatch provides two types of counts.

## Live Count

The live count represents the number of objects currently visible.

The count is smoothed over multiple frames to reduce flickering.

```text
Current view
     ↓
Detection
     ↓
Frame smoothing
     ↓
Live count
```

---

## Cumulative Count

The cumulative count tracks different objects seen during a video or live session.

Objects are tracked across frames so that the same object is not counted repeatedly.

An object must appear consistently across multiple frames before being added to the cumulative count.

The cumulative counter resets when:

- A new video is analyzed
- Live detection is restarted

---

# 📦 Stock Monitoring

ShelfWatch converts detected objects into stock information.

Example:

```text
Tomato     8     In Stock
Onion      3     Low
Milk       0     Out of Stock
Eggs       12    In Stock
```

The low-stock threshold can be configured from the sidebar.

An item is marked:

```text
In Stock
```

when its quantity is above the threshold.

```text
Low
```

when its quantity is at or below the threshold.

```text
Out of Stock
```

when its quantity reaches zero.

---

# 📱 SMS Alerts

ShelfWatch can send low-stock notifications using **Twilio**.

SMS alerts are optional.

The application can send an alert when a tracked product falls to or below the configured stock threshold.

Each alert can contain a reorder link for the detected product.

---

# 🔐 Twilio Configuration

Create:

```text
.streamlit/secrets.toml
```

with:

```toml
[twilio]
sid = "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
auth_token = "your_auth_token"
messaging_service_sid = "MGxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
to_number = "+91xxxxxxxxxx"
```

Alternatively, a Twilio phone number can be used:

```toml
[twilio]
sid = "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
auth_token = "your_auth_token"
from_number = "+1xxxxxxxxxx"
to_number = "+91xxxxxxxxxx"
```

### Important

**Never commit `secrets.toml` to GitHub.**

Add it to `.gitignore`:

```gitignore
.streamlit/secrets.toml
```

If a Twilio authentication token is accidentally exposed, revoke it and generate a new one.

---

# ⚙️ Sidebar Settings

The application provides the following settings:

| Setting | Description |
|---|---|
| **Detect: Objects / People** | Select grocery/object detection or people detection |
| **Include everyday objects** | Enables COCO objects such as bottles, cups and bowls |
| **Confidence** | Detection confidence threshold |
| **Low stock at or below** | Threshold used to determine low stock |
| **Items to track** | Products monitored for stock alerts |
| **Send SMS alerts** | Enables Twilio notifications |
| **Reset alerts** | Allows previously alerted items to trigger another alert |

---

# 🛠️ Technologies Used

- Python
- YOLOv8
- YOLOv8-OBB
- YOLO11
- Ultralytics
- OpenCV
- Streamlit
- Supervision
- NumPy
- Pandas
- Matplotlib
- PyYAML
- Twilio
- Google Colab
- Google Drive

---

# 📁 Project Structure

```text
ShelfWatch/
│
├── app.py
├── detector.py
├── alerts.py
├── config.py
├── shelfwatch.ipynb
├── requirements.txt
├── README.md
│
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml
│
└── models/
    └── best.pt
```

### Main files

| File | Purpose |
|---|---|
| `app.py` | Streamlit application interface |
| `detector.py` | Runs detection, combines models and tracks objects |
| `alerts.py` | Stock checking and Twilio SMS alerts |
| `config.py` | Model configuration and application settings |
| `shelfwatch.ipynb` | Dataset preparation, training and evaluation notebook |
| `best.pt` | Custom grocery detection model |
| `yolo11n.pt` | General COCO model |
| `.streamlit/config.toml` | Streamlit application theme |
| `.streamlit/secrets.toml` | Private Twilio credentials |

---

# 💻 Requirements

- Python **3.10 or newer**
- Webcam or RTSP/HTTP camera for live detection
- GPU recommended for faster inference and training
- Twilio account for SMS alerts (optional)

---

# 📥 Installation

Clone the repository:

```bash
git clone https://github.com/jyothiradithya2005/ShelfWatch.git
```

Enter the project:

```bash
cd ShelfWatch
```

Create a virtual environment:

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run ShelfWatch

Start the Streamlit application:

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

# ☁️ Google Colab Training

The `shelfwatch.ipynb` notebook is designed for Google Colab.

The training workflow is:

```text
Google Drive
     ↓
Dataset ZIP
     ↓
Dataset Extraction
     ↓
Class Verification
     ↓
Dataset Analysis
     ↓
80/10/10 Split
     ↓
YOLOv8m-OBB Training
     ↓
Best Model
     ↓
Validation
     ↓
Testing
     ↓
Confusion Matrix
     ↓
PR Curves
     ↓
Class-wise Metrics
     ↓
Inference
```

The notebook expects a dataset ZIP path similar to:

```text
/content/drive/MyDrive/Complete Food.v1-yolov8-obb.zip
```

Change the path if your Google Drive dataset is stored elsewhere.

---

# 🔧 Customization

Application settings can be modified in:

```text
config.py
```

Important settings include:

```text
DEFAULT_CONFIDENCE
DEFAULT_THRESHOLD
DEFAULT_TRACKED
SMOOTHING_WINDOW
REORDER_LINKS
COCO_MODEL
```

### Confidence

Lower the confidence threshold if objects are being missed.

Increase it if false detections occur.

### Smoothing Window

Controls how many frames are used to smooth live counts.

### Reorder Links

Defines the shopping/reorder link associated with each tracked product.

Products without a specific link can use a general search link.

---

# 🧪 Troubleshooting

| Problem | Solution |
|---|---|
| `ImportError` after changing code | Stop the Streamlit process with `Ctrl+C` and restart |
| `No model file at ...` | Put `best.pt` in the expected model location |
| Camera cannot open | Close applications currently using the webcam |
| Items are missed | Lower the confidence threshold and improve lighting |
| Wrong objects are detected | Increase the confidence threshold |
| Live camera is slow | Disable everyday-object detection or process fewer video frames |
| Twilio is not configured | Check `.streamlit/secrets.toml` and restart Streamlit |
| SMS failed | Check Twilio credentials, account balance and verified numbers |
| Model is slow | Use a smaller YOLO model or reduce image resolution |

---

# 🔒 Security

Never commit:

```text
.streamlit/secrets.toml
.env
*.env
API keys
Twilio Auth Tokens
GitHub Personal Access Tokens
```

Recommended `.gitignore` entries:

```gitignore
.env
*.env
.streamlit/secrets.toml
*.pt
runs/
dataset/
dataset_clean/
weights/
```

---

# 🔮 Future Improvements

Possible future improvements include:

- Real-time inventory database
- Automatic product quantity updates
- Smart shelf monitoring
- Barcode integration
- Product recognition using SKU-level models
- Improved small-object detection
- Better detection of partially occluded products
- Automated inventory reports
- Web-based analytics dashboard
- Cloud deployment
- Multi-camera monitoring
- Email notifications
- WhatsApp notifications
- Automatic stock replenishment workflows

---

# 👨‍💻 Author

**Jyothiradithya Sagiraju**

GitHub:

https://github.com/jyothiradithya2005

---

# ⭐ ShelfWatch

**AI-powered grocery detection, counting and stock monitoring.**

```text
Detect → Count → Monitor → Alert
```
