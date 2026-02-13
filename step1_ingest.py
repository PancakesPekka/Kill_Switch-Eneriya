import cv2
import os

def run_ingestion(video_path, output_folder="frames"):
    # 1. Create a folder to store the frames
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created folder: {output_folder}")

    # 2. Load the video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) # Get frames per second
    interval = int(fps * 2) # Calculate 2-second interval
    
    frame_count = 0
    saved_count = 0

    print("Extracting frames... please wait.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break # End of video
        
        # 3. Save only every 2 seconds
        if frame_count % interval == 0:
            timestamp = frame_count / fps
            filename = os.path.join(output_folder, f"frame_{saved_count:04d}_time_{int(timestamp)}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1
            
        frame_count += 1

    cap.release()
    print(f"✅ Ingestion Complete! {saved_count} frames saved in '{output_folder}'.")

if __name__ == "__main__":
    run_ingestion("test_video.mp4")