import cv2
from ultralytics import YOLO
import supervision as sv
import os

# Load YOLOv8 (v8n is fastest for real-time MOT in a hackathon)
model = YOLO('yolov8n.pt') 

def process_recovery_clip(video_path, match_time, output_name="recovery_clip.mp4"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(3)), int(cap.get(4))
    
    # BACK-TRACE LOGIC: Start 15s before the match to see the 'Loss Event'
    # End 5s after the match
    start_frame = max(0, int((match_time - 15) * fps))
    end_frame = int((match_time + 5) * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_name, fourcc, fps, (width, height))

    # Annotators for highlighting
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
        ret, frame = cap.read()
        if not ret: break

        # 1. ENHANCEMENT: Software layer to fix shadows/sunlight
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(l)
        enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

        # 2. FILTERED DETECTION: Detect only 'person' (Class 0) 
        # and your target object if it's a common class (e.g., backpack, handbag)
        results = model(enhanced, conf=0.25, classes=[0, 24, 26, 28])[0] # 0=person, 24=backpack, 26=handbag, 28=suitcase
        detections = sv.Detections.from_ultralytics(results)

        # 3. HIGHLIGHTING
        annotated = box_annotator.annotate(scene=enhanced, detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections)

        out.write(annotated)

    cap.release()
    out.release()
    return output_name