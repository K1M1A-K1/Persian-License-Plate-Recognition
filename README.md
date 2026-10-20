# Persian License Plate Detection and Recognition

A two-stage computer vision pipeline that **locates license plates with YOLO11n** and **reads Persian plate text with a multi-head CNN**. The project compares a custom residual CNN trained from scratch with a fine-tuned ResNet18, then connects the selected recognizer to the detector.

The saved experiment achieves **98.47% detection mAP@50** and **91.09% exact plate recognition accuracy** on cropped test plates. It also evaluates the complete detection-to-text pipeline, including a separate analysis of unseen plate labels.

## Implementation highlights

- **Persian-aware labels:** Unicode normalization, digit conversion, and removal of invisible formatting characters.
- **Eight-token recognition:** the middle letter or word, including `الف`, occupies one output position. This retains all **24,561** recognition samples, including **1,042** that a Unicode-length filter would exclude.
- **Two recognition approaches:** a custom CNN with residual blocks and dropout, and ImageNet-pretrained ResNet18; both use eight classification heads over a 32-token vocabulary.
- **Grouped recognition splits:** images with the same normalized plate label stay together. Every middle-token class is represented in training.
- **Validation-based selection:** save each recognizer's lowest-validation-loss checkpoint, then select between them using validation plate accuracy.
- **Complete inference:** detect, clip and crop the highest-confidence plate, apply the matching preprocessing, and decode its text. The notebook includes training curves and prediction-error examples.

For example, a multi-letter middle word is encoded as one token:

```python
tokenize_label("12الف34567")
# ["1", "2", "الف", "3", "4", "5", "6", "7"]
```

## Results

All figures below come from the executed notebook.

### Plate detection

YOLO11n evaluated on **626 test images containing 703 annotated plates**:

| Precision | Recall | mAP@50 | mAP@50–95 |
|---:|---:|---:|---:|
| 93.72% | 95.52% | **98.47%** | 76.15% |

### Recognition on cropped plates

Evaluated on **3,604 test crops** using the best validation-loss checkpoints:

| Recognizer | Token accuracy | Exact plate accuracy | Selected epoch |
|---|---:|---:|---:|
| Custom residual CNN | **98.18%** | **91.09%** | 19 / 20 |
| Fine-tuned ResNet18 | 96.85% | 84.88% | 13 / 15 |

Token accuracy measures individual plate positions; exact plate accuracy requires all eight tokens to match. The scratch CNN improves exact plate accuracy by **6.22 percentage points** in this experiment and is selected for the combined pipeline using validation performance.

### Detection-to-text pipeline

| Evaluation set | Images | Exact plate accuracy |
|---|---:|---:|
| All LPD test images | 626 | **85.62%** |
| Unseen-label subset | 181 | **65.19%** |

The detector returned no plate for **1 of 626 images**. Of the 626 ground-truth labels, **445 also occur in training or validation data** across the two stages. The overall pipeline score therefore includes seen labels. The unseen-label result is reported separately; label separation alone does not establish source-image independence. The pipeline reads one highest-confidence plate per image, while detector metrics evaluate all annotated boxes.

## Data and training

| Stage | Total images | Training | Validation | Test |
|---|---:|---:|---:|---:|
| Detection | 6,252 | 5,001 | 625 | 626 |
| Recognition | 24,561 | 17,295 | 3,662 | 3,604 |

Recognition uses approximately 70/15/15 proportions by distinct plate label. Middle-token groups with fewer than three distinct plates stay in training, so some rare classes have no test examples.

YOLO11n trains for 30 epochs at image size 640 with batch size 16. Recognition uses 128 × 64 RGB crops, batch size 64, Adam, and validation-loss learning-rate scheduling. The scratch CNN trains for 20 epochs; ResNet18 trains for 15. The saved run used **Google Colab, an NVIDIA Tesla T4, Python 3.13.15, PyTorch 2.11.0, and Ultralytics 8.4.183**.

## Run in Google Colab

1. Open the project notebook in Colab and select a GPU runtime.
2. Upload `requirements.txt` to the runtime and install dependencies:

   ```python
   %pip install -r /content/requirements.txt
   ```

3. Mount your Google Drive using the notebook's existing cell. Set `lpd_zip` and `lpr_zip` to your private copies of `LPD.zip` and `LPR.zip`.
4. Run the cells in order. Archives are extracted into the runtime; initial model-weight downloads require internet access.

The archives should contain these paths at their top level:

| Archive | Required contents |
|---|---|
| `LPD.zip` | `images/`, matching YOLO annotations in `labels/`, and `plate_labels.csv` |
| `LPR.zip` | Cropped images in `detections/` and `valid_samples.csv` |

Both CSVs use `image_path` and `label` columns. The datasets were supplied for coursework and are **not distributed with this repository**; running requires your own authorized copies.

The requirements pin the logged PyTorch and Ultralytics versions, with the [corresponding Torchvision version](https://pytorch.org/get-started/previous-versions/#v2110). Supporting dependencies use version ranges because their exact versions were not captured in the notebook.

## Saved checkpoints

Training writes `lpd_best.pt` under `/content` and `scratch_best.pt` / `resnet_best.pt` under `/content/LPR_FILES/checkpoints`. Recognition checkpoints include the model weights, token vocabulary, input size, normalization identifier, selected epoch, and validation metrics. Copy the files to Drive before ending the Colab session if you want to keep them.
