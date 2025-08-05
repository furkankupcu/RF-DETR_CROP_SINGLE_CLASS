import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Converting a tensor to a Python boolean.*")

import argparse
import cv2
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
from typing import List, Tuple

from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES

def process_and_annotate_video(
    model: RFDETRBase,
    video_path: str,
    target_class_id: int,
    threshold: float,
    padding: int,
    annotated_video_writer: cv2.VideoWriter = None,
) -> Tuple[List[Image.Image], int]:
   
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Video file could not be opened: {video_path}")
        return [], 0

    cropped_images = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"📹 Processing and annotating video: {video_path}")
    for _ in tqdm(range(total_frames), desc="Scanning Frames"):
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(frame_rgb)
        detections = model.predict(image_pil, threshold=threshold)

        best_detection_score = -1.0
        best_detection_bbox = None

        for bbox, class_id, score in zip(detections.xyxy, detections.class_id, detections.confidence):
            if class_id == target_class_id and score > best_detection_score:
                best_detection_score = score
                best_detection_bbox = bbox

        if best_detection_bbox is not None:
            x1, y1, x2, y2 = map(int, best_detection_bbox)
            
            padded_x1 = max(0, x1 - padding)
            padded_y1 = max(0, y1 - padding)
            padded_x2 = min(frame_width, x2 + padding)
            padded_y2 = min(frame_height, y2 + padding)

            padded_bbox = (padded_x1, padded_y1, padded_x2, padded_y2)
            cropped_pil = image_pil.crop(padded_bbox)
            cropped_images.append(cropped_pil)

            if annotated_video_writer:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (36, 255, 12), 2) # Green box
                class_name = COCO_CLASSES[target_class_id]
                label = f"{class_name}: {best_detection_score:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        if annotated_video_writer:
            annotated_video_writer.write(frame)

    cap.release()
    return cropped_images, total_frames


def create_video_from_images(images: List[Image.Image], output_path: str, fps: int) -> None:
    if not images:
        print("ℹ️ No images found to save for cropped video.")
        return
    max_w = max(img.width for img in images)
    max_h = max(img.height for img in images)
    target_size = (max_w, max_h)
    print(f"📏 Resizing frames to standard size for cropped video: {target_size}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, target_size)
    for pil_image in tqdm(images, desc="Creating Cropped Video"):
        padded_image = ImageOps.pad(pil_image, target_size, color="black")
        frame_cv = cv2.cvtColor(np.array(padded_image), cv2.COLOR_RGB2BGR)
        out.write(frame_cv)
    out.release()
    print(f"✅ Cropped video successfully saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Detects objects in a video. Produces two outputs from detected objects: "
                    "a 'cropped video' and an 'annotated video' with bounding boxes drawn on the original video."
    )
    parser.add_argument("--video-path", type=str, required=True, help="Path to the input video file.")
    parser.add_argument("--cropped-output-path", type=str, default="cropped_output.mp4", help="Output video containing only cropped objects.")
    parser.add_argument("--annotated-output-path", type=str, default="annotated_output.mp4", help="Full-frame video output with detection boxes drawn.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold for object detection.")
    parser.add_argument("--fps", type=int, default=25, help="Frames per second (FPS) value for output videos.")
    parser.add_argument("--class-id", type=int, default=1, help="ID of the class to detect (for COCO 'person' = 1).")
    parser.add_argument("--padding", type=int, default=20, help="Padding in pixels to add to bounding box during cropping.")
    args = parser.parse_args()

    print("🧠 Loading model...")
    model = RFDETRBase()
    model.optimize_for_inference()
    print("✨ Model optimized and ready.")

    annotated_writer = None
    try:
        cap_props = cv2.VideoCapture(args.video_path)
        if not cap_props.isOpened():
            raise FileNotFoundError(f"Input video could not be opened: {args.video_path}")
        frame_width = int(cap_props.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap_props.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_props.release()
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        annotated_writer = cv2.VideoWriter(args.annotated_output_path, fourcc, args.fps, (frame_width, frame_height))
        print(f"✍️ Annotated video will be saved to: {args.annotated_output_path}")

        cropped_pil_images, total_input_frames = process_and_annotate_video(
            model=model,
            video_path=args.video_path,
            target_class_id=args.class_id,
            threshold=args.threshold,
            padding=args.padding,
            annotated_video_writer=annotated_writer
        )

        total_output_frames = len(cropped_pil_images)
        target_class_name = COCO_CLASSES[args.class_id]
        print("\n--- Process Summary ---")
        print(f"📊 Input Video Total Frame Count: {total_input_frames}")
        print(f"🎯 Found '{target_class_name}' Object Count (Cropped Video Frame Count): {total_output_frames}")
        if total_input_frames > 0:
            detection_ratio = (total_output_frames / total_input_frames) * 100
            print(f"📈 Detection Ratio: {detection_ratio:.2f}%")
        print("-------------------\n")

        create_video_from_images(
            images=cropped_pil_images,
            output_path=args.cropped_output_path,
            fps=args.fps,
        )

    except FileNotFoundError:
        print(f"❌ Error: Specified video file not found: {args.video_path}")
    except KeyError:
        print(f"❌ Error: Invalid Class ID: {args.class_id}.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if annotated_writer:
            annotated_writer.release()
            print(f"✅ Annotated video successfully saved: {args.annotated_output_path}")

if __name__ == "__main__":
    main()