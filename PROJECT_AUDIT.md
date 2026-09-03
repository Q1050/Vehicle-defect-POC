# Vehicle Defect YOLO: Current-State Audit

Audited: 2026-08-19

No existing source files were modified for this audit.

## 1. Project Structure

```text
vehicle-defect-yolo/
├── app.py
├── requirements.txt
├── .env                         # Gemini API key, sensitive
├── data/
│   └── chat_state.json          # Persisted chats, attachments, reports
├── modules/
│   ├── __init__.py
│   ├── attachments.py
│   ├── chat_storage.py
│   └── media_analysis.py
├── models/
│   └── best.pt                  # Fallback trained model
├── dataset/
│   ├── data.yaml
│   ├── README.roboflow.txt
│   ├── yolo11n.pt               # Base YOLOv11 model
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── runs/                    # Generated dataset artifacts
├── runs/
│   └── detect/
│       ├── train_v2/            # Main generated training run
│       └── dataset/smoke_test/  # Generated run
├── venv/                        # Excluded
└── __pycache__/                 # Excluded
```

The Roboflow README reports 24 dataset images.

## 2. Main Entry Point

`app.py` is the sole application entry point. It is intended to run with:

```text
streamlit run app.py
```

There is no separate backend, API server, CLI, or `main()` function.

## 3. Important Python Files

- `app.py`: Builds the Streamlit UI, manages chats, loads YOLO, handles uploads, triggers analysis, renders results, and persists state.
- `modules/media_analysis.py`: Performs YOLO image inference, draws annotations, builds reports, calls Gemini, and generates chat responses.
- `modules/attachments.py`: Converts Streamlit uploads into image/video payloads and identifies attachment types.
- `modules/chat_storage.py`: Serializes and deserializes chat messages, media, annotated images, and reports into `data/chat_state.json`.
- `modules/__init__.py`: Package marker only.

## 4. YOLO Model and Weights

`app.py` loads models in this order:

1. `runs/detect/train_v2/weights/best.pt`
2. `models/best.pt`
3. `dataset/yolo11n.pt`
4. `yolov8n.pt` as a final Ultralytics fallback

The primary configured model is:

```text
runs/detect/train_v2/weights/best.pt
```

Training used `dataset/yolo11n.pt` as the pretrained base model. The five configured classes are:

```text
broken_belt
dashboard_indicator
oil_leak
rust
tire_wear
```

## 5. Inference

For images:

1. Uploaded bytes are opened as an RGB PIL image.
2. The image is converted to a NumPy array.
3. YOLO inference runs with confidence threshold `0.65`.
4. Bounding boxes, class IDs, confidences, and coordinates are extracted.
5. A custom annotated image is generated.
6. A problem statement is built from detected labels and confidence scores.
7. Up to three detection crops may be sent to Gemini.

For videos, YOLO is not run on frames. The complete video is sent to Gemini for analysis.

## 6. LLM Integration

Gemini is integrated through `google-genai` in `modules/media_analysis.py`.

It is used for:

- Detection-crop interpretation for images
- Whole-video analysis
- Chat-only replies
- Post-analysis summaries using conversation and report context

The configured model is `gemini-3.5-flash`.

Without a valid key or package, the application falls back to local summaries and explanatory messages.

## 7. Dependencies and Imports

Declared in `requirements.txt`:

- `ultralytics`
- `streamlit`
- `opencv-python`
- `pillow`
- `numpy`
- `google-genai`
- `python-dotenv`

Standard-library imports include `html`, `io`, `os`, `tempfile`, `base64`, `json`, `uuid`, and `collections.Counter`.

Third-party imports include PIL, NumPy, Streamlit, python-dotenv, and Ultralytics. `google.genai` is imported conditionally. `opencv-python` is declared but not directly imported by project source.

## 8. Configuration and Secrets

- `.env` is loaded with `load_dotenv()`.
- The application reads `GEMINI_API_KEY`.
- A plaintext Gemini API key is currently present in `.env`; the value is intentionally not reproduced here. Rotate it immediately if the workspace or repository has been shared or exposed.
- `dataset/data.yaml` defines dataset paths and class names.
- `runs/detect/train_v2/args.yaml` records training configuration.
- `data/chat_state.json` is runtime data, not configuration.

## 9. Unfinished, Risky, or Broken Areas

- No tests were found.
- Dependency versions are unpinned.
- The `yolov8n.pt` fallback is a generic COCO model, not a vehicle-defect model. Its labels will not match the project classes if it is used.
- Videos bypass YOLO and depend entirely on Gemini availability.
- Chat state stores complete uploaded media and generated images as base64 in one JSON file, which can grow very large.
- Persisted chat state contains user media and previous AI-generated safety advice.
- Several model-generated values are inserted into HTML with `unsafe_allow_html=True` without consistent escaping.
- The dataset is very small. The recorded final training metrics are weak: approximately `mAP50 = 0.134`, `mAP50-95 = 0.043`, precision `0.242`, and recall `0.210`.
- Model and state paths are relative to the current working directory.
- There is no model version validation, database, authentication, multi-user isolation, or upload-size policy.
- No explicit `TODO` or `FIXME` markers were found.

## 10. User Flow

1. Launch Streamlit with `app.py`.
2. Load `.env` and the cached YOLO model.
3. Load existing chats from `data/chat_state.json`, or create a new chat.
4. Select or create a chat thread.
5. Enter an issue description and optionally upload images or one video.
6. Store the message in the selected chat.
7. Run YOLO on image attachments or Gemini on video attachments.
8. Optionally generate Gemini interpretations and conversational summaries.
9. Display annotated media, detections, confidence scores, and summaries.
10. Persist the conversation, media, and reports back to `data/chat_state.json`.

## 11. Most Important Files to Inspect Next

1. `app.py`
2. `modules/media_analysis.py`
3. `modules/chat_storage.py`
4. `modules/attachments.py`
5. `requirements.txt`
