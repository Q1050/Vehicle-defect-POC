# Vehicle Defect YOLO

Training workspace and proof-of-concept diagnostic packages used by AutoAssist.

Model weights, generated datasets, local state, and training runs are intentionally excluded from Git. The canonical AutoAssist image checkpoint is distributed separately through Google Drive:

`Use the shared drive to access this`

After downloading it, either place it at:

```text
runs/detect/vehicle_defect_v3/weights/best.pt
```

or configure the backend's `AUTOASSIST_IMAGE_MODEL_PATH` to its absolute or backend-relative location. See the external model handoff manifest for the expected filename, size, and SHA-256 checksum.

The packages under `pocs/` provide deterministic audio, video, realtime-audio, evidence-fusion, and symptom-extraction implementations. They currently remain sibling source dependencies of `autoassist-backend`; they do not require additional trained weight files.
"# Vehicle-defect-POC" 
