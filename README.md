# 🦁 Wildlife Camera Trap Detector

Wildlife detector for Costa Rica and Central America using **YOLOv8n** trained via transfer learning on the [WCS Camera Traps](https://lila.science/datasets/wcscameratraps/) dataset.

---

## 📋 Project Description

This project implements a complete Deep Learning pipeline for **object detection** of wildlife in camera trap images. The model detects 12 species relevant to Costa Rica and Central America, and can process real camera trap videos to visualize detections with bounding boxes, labels, and confidence scores.

### Detected species

| ID | Scientific name | Common name |
|----|-------------------|--------------|
| 0  | *Crax rubra* | Great Curassow |
| 1  | *Dasyprocta punctata* | Central American Agouti |
| 2  | *Leopardus pardalis* | Ocelot |
| 3  | *Mazama temama* | Central American Red Brocket |
| 4  | *Meleagris ocellata* | Ocellated Turkey |
| 5  | *Nasua narica* | White-nosed Coati |
| 6  | *Odocoileus virginianus* | White-tailed Deer |
| 7  | *Panthera onca* | Jaguar |
| 8  | *Pecari tajacu* | Collared Peccary |
| 9  | *Puma concolor* | Puma |
| 10 | *Tapirus bairdii* | Baird's Tapir |
| 11 | *Tayassu pecari* | White-lipped Peccary |

---

## 🎬 Inference Video

The rendered video with wildlife detections is not included in this repo due to size constraints.

You can access it here:

👉 [wildlife_detected.mp4 — View on Google Drive](https://drive.google.com/file/d/1oqCN0tsEjS6VG4n59_XvSuFZRR0JBEzP/view?usp=sharing)

**Description:** A video from the Guanacaste Wildlife Monitoring channel processed with the trained model. Shows the first 2 minutes with bounding boxes, species label, and confidence percentage for each detection.

---

## 🏗️ Project Architecture

```
wildlife-detector/
│
├── 01_data_preparation.ipynb   ← Dataset preparation and conversion
├── 02_training.ipynb           ← Training on Colab with GPU
├── 03_evaluation.ipynb         ← Metrics and results visualization
├── 04_video_inference.ipynb    ← Inference and video rendering
│
└── README.md                   ← This file
```

> The four notebooks are self-contained and meant to be run in order (01 → 04) on Google Colab. Each notebook generates the data, model weights, or outputs the next one depends on — see the step-by-step guide below.

---

## ⚙️ Requirements

### Software
- Python 3.10+
- Google Colab (recommended for training with T4 GPU)
- VS Code (recommended for local development)

### Python dependencies
```
ultralytics>=8.0
requests
tqdm
pillow
opencv-python-headless
yt-dlp
matplotlib
numpy
pyyaml
```

---

## 🚀 Step-by-Step Execution Guide

### Step 1 — Get the dataset

Download the dataset JSON file:

```
wcs_20220205_bboxes_latam_animals.json
```

Place it in the project's `data/` folder.

---

### Step 2 — Prepare the dataset (`01_data_preparation.ipynb`)

**Where to run it:** Google Colab (or locally with a good internet connection)

**What it does:**
- Reads the JSON in COCO format
- Filters the 12 target Central American species
- Selects up to 300 images per species (~3,600 images total)
- Downloads the images from LILA Science's public servers
- Converts bounding boxes from COCO format to YOLO format
- Splits the dataset into train/val/test (70%/20%/10%)
- Generates the `data.yaml` file

**COCO → YOLO conversion:**
```
COCO : [x_min, y_min, width, height]  (absolute pixels)
YOLO : [class x_center y_center width height]  (normalized 0-1)

x_center = (x_min + width/2)  / img_width
y_center = (y_min + height/2) / img_height
```

**Estimated time:** 20–40 minutes

---

### Step 3 — Train the model (`02_training.ipynb`)

**Where to run it:** Google Colab with T4 GPU

> ⚠️ **Enable GPU:** Runtime → Change runtime type → T4 GPU

**What it does:**
- Loads pretrained `yolov8n.pt` weights (COCO, 80 classes, ~6M parameters)
- Fine-tunes on the wildlife dataset (12 classes)
- Uses data augmentation: mosaic, horizontal flip, HSV jitter, scaling, rotation
- Applies early stopping with patience=15 epochs
- Saves the best model as `best.pt`

**Architecture — YOLOv8n:**
- Backbone: enhanced CSPDarknet
- Neck: PANet with FPN
- Head: decoupled head (anchor-free)
- Parameters: ~3.2M
- Speed: ~80 FPS on T4

**Estimated time:** 1–3 hours (50 epochs)

---

### Step 4 — Evaluate the model (`03_evaluation.ipynb`)

**Where to run it:** Google Colab (GPU not strictly required)

**What it does:**
- Runs inference on the test split (previously unseen images)
- Calculates mAP50, Precision, and Recall per class
- Generates a Confusion Matrix
- Visualizes predictions on sample images

**Key metrics:**
- **mAP50:** Main metric — mean average precision at IoU≥0.5
- **Precision:** TP / (TP + FP) — how reliable the detections are
- **Recall:** TP / (TP + FN) — what percentage of animals it detects

---

### Step 5 — Process video (`04_video_inference.ipynb`)

**Where to run it:** Google Colab with GPU (recommended)

**What it does:**
1. Downloads a video from the [Guanacaste Wildlife Monitoring](https://www.youtube.com/@guanacastewildlifemonitori8194) channel using `yt-dlp`
2. Processes each frame with YOLOv8n
3. Renders bounding boxes with label and confidence
4. Exports the final video as `wildlife_detected.mp4`

**To use with your own video:**
1. Go to the YouTube channel
2. Choose a video (recommended: at least 1 minute long)
3. Copy the URL
4. Paste it into the `YOUTUBE_URL` variable in the notebook

---

## 🧠 Design Decisions

### Why YOLOv8n (nano)?
- Smallest model in the YOLOv8 family (~3.2M parameters)
- Trains on free Colab (T4, 16GB VRAM) without memory issues
- High inference speed for video processing in a reasonable time
- Accurate enough for the project's goal

### Why only Guatemala from the dataset?
- The Latam dataset includes Bolivia, Guatemala, Venezuela, and Ecuador
- Guatemala has the wildlife most similar to Costa Rica (same biogeographic region)
- Baird's tapir, jaguar, puma, coati, and white-lipped peccary are present in both countries

### Why 300 images per class?
- Balance between variety and training time
- Enough for transfer learning to work well
- Manageable for free Colab (disk space)

### Why transfer learning from COCO?
- COCO has 80 classes with millions of images
- COCO's animal classes (dogs, cats, cows, bears) share visual features with wildlife species
- Transfer learning reduces training time from hours/days to 1-3 hours
- Better performance with small datasets than training from scratch

---

## 📊 Results

Training over 30 epochs on ~3,600 images with YOLOv8n:

| Metric | Result |
|---------|-----------|
| mAP50 | 0.855 (85.5%) |
| mAP50-95 | 0.679 (67.9%) |
| Precision | 0.864 (86.4%) |
| Recall | 0.802 (80.2%) |

**Best-performing classes:** Central American Agouti (92.2%), Puma (92.1%), Central American Red Brocket (88.8%)
**Hardest class:** White-nosed Coati (77.9%) — multiple individuals per image

**Known limitations:**
- The model only detects the 12 trained species
- Dataset is from Guatemala, not exactly Costa Rica
- ~300 images per class (relatively small dataset)

---

## 📚 References

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [WCS Camera Traps Dataset — LILA Science](https://lila.science/datasets/wcscameratraps/)
- [COCO Camera Traps Format](https://github.com/agentmorris/MegaDetector/blob/main/megadetector/data_management/README.md)
- [Guanacaste Wildlife Monitoring (YouTube)](https://www.youtube.com/@guanacastewildlifemonitori8194)
