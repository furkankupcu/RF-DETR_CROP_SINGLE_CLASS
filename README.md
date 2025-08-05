# RF-DETR Video Object Detection Tool

This project uses the RF-DETR model to detect specific objects in a video. It generates two outputs based on the detected objects:
- 🔳 **Cropped Video**: A video composed only of cropped frames containing the detected objects.
- 🟥 **Annotated Video**: The original video with bounding boxes drawn around detected objects.

---

## 🛠️ Requirements

You need the following Python packages installed:

```bash
pip install -r requirements.txt
```
---

python main.py \
  --video-path "videos/input_video.mp4" \
  --cropped-output-path "outputs/cropped_output.mp4" \
  --annotated-output-path "outputs/annotated_output.mp4" \
  --class-id 1 \
  --threshold 0.5 \
  --fps 25 \
  --padding 20


---

| Argument                  | Description                                           |
| ------------------------- | ----------------------------------------------------- |
| `--video-path`            | Path to the input video file (required)               |
| `--cropped-output-path`   | Output path for the cropped objects video             |
| `--annotated-output-path` | Output path for the annotated full-frame video        |
| `--class-id`              | COCO class ID to detect (e.g. person = 1)             |
| `--threshold`             | Confidence threshold for detection (default: 0.5)     |
| `--fps`                   | Frames per second for output videos                   |
| `--padding`               | Padding in pixels around bounding boxes when cropping |

```bash
outputs/
├── cropped_output.mp4       # Video with only cropped detected objects
└── annotated_output.mp4     # Original video with bounding boxes drawn
```






https://github.com/user-attachments/assets/008efa1d-0ed2-446f-b92e-35210cc17cbc





https://github.com/user-attachments/assets/48c11ffb-eaf0-4680-8fde-f061ef45fdb0




