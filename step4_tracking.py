import cv2
from ultralytics import YOLO
import supervision as sv
import os

# Load the model
model = YOLO('yolov8n.pt') 

def process_recovery_clip(video_path, match_time, output_name="recovery_clip.mp4"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(3)), int(cap.get(4))
    
    # Temporal Scale: Start 15s before to catch the initial presence
    start_frame = max(0, int((match_time - 15) * fps))
    end_frame = int((match_time + 5) * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_name, fourcc, fps, (width, height))

    # We use a specific color for the "Target" to differentiate it
    box_annotator = sv.BoxAnnotator(color=sv.Color.from_hex("#FF0000")) 
    label_annotator = sv.LabelAnnotator(color=sv.Color.from_hex("#FF0000"))

    while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
        ret, frame = cap.read()
        if not ret: break

        # Enhancement Layer for Adverse Conditions (Sunlight/Shadows)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(l)
        enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

        # 1. Targeted Detection: Limit classes to common lost items
        # 24: backpack, 26: handbag, 28: suitcase, 39: bottle
        results = model(enhanced, conf=0.20, classes=[24, 26, 28, 39])[0]
        detections = sv.Detections.from_ultralytics(results)

        # 2. ISOLATION LOGIC: If multiple bags are found, 
        # for the MVP we will highlight the one with the highest confidence
        if len(detections) > 0:
            # Sort by confidence and take the top 1
            best_detection = detections[0:1] 
            
            # 3. Highlighting only the isolated object
            annotated = box_annotator.annotate(scene=enhanced, detections=best_detection)
            annotated = label_annotator.annotate(scene=annotated, detections=best_detection)
            out.write(annotated)
        else:
            out.write(enhanced)

    cap.release()
    out.release()
    return output_name