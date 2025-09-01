import os
import glob
import cv2
import time
import pickle

class Endgame:
    def __init__(self):
        self.root = 'difftactile/manual_or_experimental_data/endgame'
        self.dir = '20250901-131547'

    def annotate(self):
        input_dir = os.path.join(self.root, f"{self.dir}_dilated")
        output_dir = os.path.join(self.root, f"{self.dir}_annotations")
        os.makedirs(output_dir, exist_ok=True)

        avi_files = sorted(glob.glob(os.path.join(input_dir, "*.avi")))
        if not avi_files:
            print("No videos found.")
            return

        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)]  # BGR: red, green, blue, magenta

        # Load or initialize annotations
        annotations = {}
        for avi_path in avi_files:
            base = os.path.splitext(os.path.basename(avi_path))[0]
            pkl_path = os.path.join(output_dir, f"{base}.pkl")
            if os.path.exists(pkl_path):
                with open(pkl_path, "rb") as f:
                    annotations[avi_path] = pickle.load(f)
            else:
                cap = cv2.VideoCapture(avi_path)
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                annotations[avi_path] = [[] for _ in range(n_frames)]

        # State variables
        video_idx = 0
        frame_idx = 0

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                avi_path = avi_files[video_idx]
                frame_annots = annotations[avi_path][frame_idx]
                if len(frame_annots) < 4:
                    annotations[avi_path][frame_idx].append([x, y])
                    print(f"Added point {len(frame_annots)}/{4}")
                else:
                    print("Maximum 4 points allowed.")

        cv2.namedWindow("Annotator")
        cv2.setMouseCallback("Annotator", mouse_callback)

        while True:
            avi_path = avi_files[video_idx]
            cap = cv2.VideoCapture(avi_path)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idx = max(0, min(frame_idx, n_frames - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                print(f"Failed to read frame {frame_idx} from {avi_path}")
                break

            display = frame.copy()

            # Draw annotations for current frame
            frame_annots = annotations[avi_path][frame_idx]
            for idx, pt in enumerate(frame_annots):
                color = colors[idx % len(colors)]
                cv2.circle(display, (int(pt[0]), int(pt[1])), 6, color, -1)

            # Overlay text
            text = f"Video {video_idx+1}/{len(avi_files)} | Frame {frame_idx+1}/{n_frames} | Annotations {len(frame_annots)}/4"
            cv2.putText(display, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv2.imshow("Annotator", display)
            key = cv2.waitKey(20) & 0xFF  # refresh faster to show new points immediately

            if key == ord("q"):
                # Save all annotations before exit
                for avi_path in avi_files:
                    base = os.path.splitext(os.path.basename(avi_path))[0]
                    pkl_path = os.path.join(output_dir, f"{base}.pkl")
                    with open(pkl_path, "wb") as f:
                        pickle.dump(annotations[avi_path], f)
                break
            elif key == ord("m"):
                video_idx = (video_idx + 1) % len(avi_files)
                frame_idx = 0
            elif key == ord("n"):
                video_idx = (video_idx - 1) % len(avi_files)
                frame_idx = 0
            elif key == ord("k"):
                frame_idx += 1
            elif key == ord("j"):
                frame_idx -= 1
            elif key == ord("d"):
                annotations[avi_path][frame_idx] = []
                print("Deleted annotations for this frame.")
            elif key == ord("p"):
                for avi_path in avi_files:
                    base = os.path.splitext(os.path.basename(avi_path))[0]
                    pkl_path = os.path.join(output_dir, f"{base}.pkl")
                    with open(pkl_path, "wb") as f:
                        pickle.dump(annotations[avi_path], f)
                print("Annotations saved.")

        cv2.destroyAllWindows()
        for _ in range(10):
            cv2.waitKey(1)
            time.sleep(0.1)
