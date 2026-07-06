import os
import cv2
import glob
import random
import argparse
from tqdm import tqdm

# Import FaceExtractor from the detection module
from detection.face_extractor import FaceExtractor

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def extract_faces_from_video(video_path, extractor, num_frames=5):
    """
    Samples num_frames evenly from a video and extracts the largest face from each.
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        return []
    
    # Calculate frame indices to sample evenly across the video
    indices = [int(i * frame_count / num_frames) for i in range(num_frames)]
    
    faces = []
    for idx in indices:
        # Prevent seeking past the very last frame
        safe_idx = min(idx, frame_count - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, safe_idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Extract face
        detected = extractor.extract_faces(frame)
        if detected:
            # We only want the main subject's face, so get the largest one
            largest_face = None
            max_area = 0
            for face_img, (x1, y1, x2, y2) in detected:
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    largest_face = face_img
            
            if largest_face is not None:
                faces.append(largest_face)
                
    cap.release()
    return faces

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset by extracting faces from Celeb-DF videos")
    parser.add_argument("--archive_dir", type=str, default="archive", help="Path to archive directory containing the videos")
    parser.add_argument("--output_dir", type=str, default="dataset", help="Output directory for the extracted face crops")
    parser.add_argument("--frames_per_video", type=int, default=5, help="Number of frames to sample and extract per video")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio (e.g. 0.2 means 20% validation)")
    parser.add_argument("--max_videos", type=int, default=None, help="Maximum number of videos to process per category (for testing)")
    
    args = parser.parse_args()
    
    print("Initializing Face Extractor (this may take a moment)...")
    extractor = FaceExtractor(detector_type="mtcnn")
    
    # Setup PyTorch ImageFolder style directories
    train_real = os.path.join(args.output_dir, "train", "real")
    train_fake = os.path.join(args.output_dir, "train", "fake")
    val_real = os.path.join(args.output_dir, "val", "real")
    val_fake = os.path.join(args.output_dir, "val", "fake")
    
    for d in [train_real, train_fake, val_real, val_fake]:
        ensure_dir(d)
        
    # Map archive folder names to their true labels
    sources = {
        "Celeb-real": "real",
        "YouTube-real": "real",
        "Celeb-synthesis": "fake"
    }
    
    total_extracted = 0
    
    for folder, label in sources.items():
        folder_path = os.path.join(args.archive_dir, folder)
        if not os.path.exists(folder_path):
            print(f"Skipping {folder_path} (Directory not found)")
            continue
            
        videos = glob.glob(os.path.join(folder_path, "*.mp4"))
        if args.max_videos is not None:
            videos = videos[:args.max_videos]
            
        print(f"\nProcessing {len(videos)} videos in '{folder}' as '{label}'...")
        
        for video_path in tqdm(videos):
            # Decide whether this entire video goes to train or val set
            is_val = random.random() < args.val_split
            
            if label == "real":
                out_dir = val_real if is_val else train_real
            else:
                out_dir = val_fake if is_val else train_fake
                
            vid_name = os.path.splitext(os.path.basename(video_path))[0]
            
            # Extract faces
            faces = extract_faces_from_video(video_path, extractor, num_frames=args.frames_per_video)
            
            # Save crops to disk
            for i, face_img in enumerate(faces):
                out_path = os.path.join(out_dir, f"{vid_name}_face_{i}.jpg")
                cv2.imwrite(out_path, face_img)
                total_extracted += 1
                
    print(f"\n✅ Done! Extracted {total_extracted} faces into the '{args.output_dir}' directory.")
    print(f"You can now run: python fine_tune.py --data_dir {args.output_dir} --model xception")

if __name__ == "__main__":
    main()
