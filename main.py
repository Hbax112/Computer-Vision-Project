import cv2
import random
import time
import threading
import subprocess
import os


class FocusGuardian:
    def __init__(self):

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        print("Using OpenCV Haar Cascades for face tracking")

        self.warnings = [
            "Attention drifting detected.",
            "Back to focus — this won’t stop otherwise.",
            "Nice try. Eyes back on screen.",
            "Focus mode is calling.",
            "Return your gaze to proceed.",
            "Focus restored = video stops.",
        ]

        self.last_warning_time = 0.0
        self.warning_cooldown = 3.0
        self.current_warning = ""

        self.video_path = "rickroll.mp4"
        self.video_process = None
        self.video_playing = False

        self.suspect_start_time = None
        self.SUSPECT_DELAY = 3.5  

        self.camera_index = 1
        self.backend = cv2.CAP_DSHOW


    def detect_attention_drift(self, frame, gray) -> bool:
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            roi_gray = gray[y:y + int(h * 0.6), x:x + w]
            roi_color = frame[y:y + int(h * 0.6), x:x + w]

            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
            score = 0

            face_center_y = y + h // 2
            frame_h = frame.shape[0]
            face_ratio = face_center_y / max(frame_h, 1)

            if face_ratio > 0.58:
                score += 2
            elif face_ratio > 0.52:
                score += 1

            if h / max(w, 1) < 1.1:
                score += 1

            if len(eyes) >= 2:
                eye_y = [y + ey + eh // 2 for (_, ey, _, eh) in eyes]
                avg_eye_y = sum(eye_y) / len(eye_y)
                eye_ratio = (avg_eye_y - y) / max(h, 1)

                if eye_ratio > 0.6:
                    score += 2
                elif eye_ratio > 0.52:
                    score += 1

                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            else:
                score += 1

            return score >= 3

        return False


    def start_rickroll(self):
        if self.video_playing or not os.path.exists(self.video_path):
            return

        self.video_playing = True

        def play():
            vlc_path = r"C:\Program Files\VideoLAN\VLC\vlc.exe"
            if os.path.exists(vlc_path):
                self.video_process = subprocess.Popen(
                    [
                        vlc_path,
                        "--loop",
                        "--no-video-title-show",
                        "--fullscreen",
                        self.video_path,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                os.startfile(self.video_path)

        threading.Thread(target=play, daemon=True).start()

    def stop_rickroll(self):
        if not self.video_playing:
            return

        try:
            if self.video_process:
                self.video_process.terminate()
        except Exception:
            pass

        self.video_process = None
        self.video_playing = False

    def show_warning(self, frame):
        now = time.time()
        if now - self.last_warning_time > self.warning_cooldown:
            self.current_warning = random.choice(self.warnings)
            self.last_warning_time = now

        overlay = frame.copy()
        h, w = frame.shape[:2]

        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        cv2.putText(
            frame,
            "FOCUS LOST",
            (20, 45),
            cv2.FONT_HERSHEY_DUPLEX,
            1.0,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            self.current_warning,
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )

    def run(self):
        cap = cv2.VideoCapture(self.camera_index, self.backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        for _ in range(30):
            cap.read()
            time.sleep(0.01)

        print("Focus Guardian running")
        print("Press 'q' to quit")

        try:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue

                if frame.mean() < 5:
                    continue

                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                raw = self.detect_attention_drift(frame, gray)
                now = time.time()

                if raw:
                    if self.suspect_start_time is None:
                        self.suspect_start_time = now
                else:
                    self.suspect_start_time = None
                    self.stop_rickroll()

                is_suspect = self.suspect_start_time is not None
                is_attention_lost = (
                    is_suspect and (now - self.suspect_start_time) >= self.SUSPECT_DELAY
                )

                if is_attention_lost:
                    self.show_warning(frame)
                    self.start_rickroll()
                elif is_suspect:
                    remaining = self.SUSPECT_DELAY - (now - self.suspect_start_time)
                    cv2.putText(
                        frame,
                        f"Attention drifting… ({remaining:.1f}s)",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 165, 255),
                        2,
                    )
                else:
                    cv2.putText(
                        frame,
                        "Focused — keep going!",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                cv2.imshow("Focus Guardian", frame)

                if cv2.waitKey(5) & 0xFF == ord("q"):
                    break

        finally:
            self.stop_rickroll()
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    FocusGuardian().run()
