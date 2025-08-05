# RF_DETR_CROP_SINGLE_PERSON

# RF-DETR Video Object Detection Tool

This project uses the RF-DETR model to detect specific objects in a video. It generates two outputs based on the detected objects:
- 🔳 **Cropped Video**: A video composed only of cropped frames containing the detected objects.
- 🟥 **Annotated Video**: The original video with bounding boxes drawn around detected objects.

---

python main.py \
  --video-path "videos/input_video.mp4" \
  --cropped-output-path "outputs/cropped_output.mp4" \
  --annotated-output-path "outputs/annotated_output.mp4" \
  --class-id 1 \
  --threshold 0.5 \
  --fps 25 \
  --padding 20

Argument	Description

--video-path	(Zorunlu) Girdi video dosyasının yolu.
--cropped-output-path	Kırpılmış nesneleri içeren videonun çıktı yolu.
--annotated-output-path	İşaretlenmiş tam kare videonun çıktı yolu.
--class-id	Tespit edilecek COCO sınıf ID'si (örneğin, 1 kişi anlamına gelir).
--threshold	Tespit için güven eşiği (varsayılan: 0.5).
--fps	Çıktı videoları için saniye başına kare sayısı (FPS).
--padding	Kırpma sırasında sınırlayıcı kutu etrafındaki piksel cinsinden dolgu.

outputs/
├── cropped_output.mp4       # Video with only cropped detected objects
└── annotated_output.mp4     # Original video with bounding boxes drawn
