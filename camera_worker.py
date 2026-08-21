import time
import threading
from collections import defaultdict, Counter
from datetime import datetime

import cv2
from ultralytics import YOLO

import db

CLASS_NAMES = {0: "Bus", 1: "Car", 2: "Motorcycle", 3: "Truck"}
CLASSES = [0, 1, 2, 3]
INTERVAL_SECONDS = 7 * 60  # roll up volume every 7 minutes
MIN_FRAME_VOTING = 4
RECONNECT_DELAY_SECONDS = 5


def resolve_stream_url(youtube_url: str) -> str:
    """Resolve the direct (HLS) stream URL of a YouTube live URL using yt-dlp."""
    import yt_dlp

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "format": "best",
        # Safari cookies (logged-in YouTube account) get us past the
        # "sign in to confirm you're not a bot" check.
        # The player client is left at its default (tv/web): the "android" client
        # alone no longer returns video formats (it lacks the required PO token).
        "cookiesfrombrowser": ("safari",),
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info["url"]


class CameraWorker:
    """Capture + tracking + counting for a single live CCTV feed, run on a background thread."""

    def __init__(self, name, youtube_url, model_path, roi_margin=0.1):
        self.name = name
        self.youtube_url = youtube_url
        self.model_path = model_path
        self.roi_margin = roi_margin

        self._lock = threading.Lock()
        self._thread = None
        self._running = False

        self.latest_frame = None
        self.status = "starting"
        self.error_message = ""

        self.total_counts = {c: 0 for c in CLASSES}
        self.interval_counts = {c: 0 for c in CLASSES}
        self.window_start = time.time()
        self.history = []

        self._id_counted = set()
        self._class_history = defaultdict(list)

    def start(self):
        if self._thread is not None:
            return
        db.init_db()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _run(self):
        try:
            model = YOLO(self.model_path)
        except Exception as e:
            with self._lock:
                self.status = "error"
                self.error_message = f"Failed to load model: {e}"
            return

        while self._running:
            cap = None
            try:
                with self._lock:
                    self.status = "connecting"
                stream_url = resolve_stream_url(self.youtube_url)
                cap = cv2.VideoCapture(stream_url)

                if not cap.isOpened():
                    raise RuntimeError("Cannot open stream")

                with self._lock:
                    self.status = "running"
                    self.error_message = ""

                fail_count = 0
                while self._running:
                    ok, frame = cap.read()
                    if not ok:
                        fail_count += 1
                        if fail_count > 20:
                            raise RuntimeError("Stream connection lost")
                        time.sleep(0.2)
                        continue
                    fail_count = 0

                    self._process_frame(frame, model)
                    self._maybe_roll_window()
            except Exception as e:
                with self._lock:
                    self.status = "reconnecting"
                    self.error_message = str(e)
                time.sleep(RECONNECT_DELAY_SECONDS)
            finally:
                if cap is not None:
                    cap.release()

    def _process_frame(self, frame, model):
        tinggi, lebar = frame.shape[:2]
        y1, y2 = int(tinggi * self.roi_margin), int(tinggi * (1 - self.roi_margin))
        x1, x2 = int(lebar * self.roi_margin), int(lebar * (1 - self.roi_margin))
        area_fokus = frame[y1:y2, x1:x2]

        hasil = model.track(
            source=area_fokus,
            classes=CLASSES,
            conf=0.40,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        frame[y1:y2, x1:x2] = hasil[0].plot()
        boxes = hasil[0].boxes
        id_saat_ini = set()

        with self._lock:
            if boxes.id is not None:
                id_tracking = boxes.id.cpu().numpy().astype(int)
                kelas_objek = boxes.cls.cpu().numpy().astype(int)
                id_saat_ini = set(id_tracking)

                for i in range(len(id_tracking)):
                    id_obj = id_tracking[i]
                    id_kelas = kelas_objek[i]
                    self._class_history[id_obj].append(id_kelas)

                    if len(self._class_history[id_obj]) >= MIN_FRAME_VOTING and id_obj not in self._id_counted:
                        frame_terakhir = self._class_history[id_obj][-5:]
                        kelas_akhir = Counter(frame_terakhir).most_common(1)[0][0]
                        self._register_count(kelas_akhir)
                        self._id_counted.add(id_obj)

            for id_obj in list(self._class_history.keys()):
                if id_obj not in id_saat_ini and id_obj not in self._id_counted:
                    kelas_akhir = self._class_history[id_obj][-1]
                    self._register_count(kelas_akhir)
                    self._id_counted.add(id_obj)

            # drop history for IDs already counted and no longer visible, so memory stays bounded
            for id_obj in list(self._class_history.keys()):
                if id_obj in self._id_counted and id_obj not in id_saat_ini:
                    del self._class_history[id_obj]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "Detection area", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            self.latest_frame = frame

    def _register_count(self, kelas):
        self.total_counts[kelas] += 1
        self.interval_counts[kelas] += 1

    def _maybe_roll_window(self):
        with self._lock:
            elapsed = time.time() - self.window_start
            if elapsed >= INTERVAL_SECONDS:
                entry = {
                    "Camera": self.name,
                    "Start": datetime.fromtimestamp(self.window_start).strftime("%Y-%m-%d %H:%M:%S"),
                    "End": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                entry.update({CLASS_NAMES[c]: v for c, v in self.interval_counts.items()})
                entry["Total"] = sum(self.interval_counts.values())
                self.history.append(entry)
                db.insert_history(entry)
                self.interval_counts = {c: 0 for c in CLASSES}
                self.window_start = time.time()

    def snapshot(self):
        with self._lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
            return {
                "status": self.status,
                "error": self.error_message,
                "frame": frame,
                "total_counts": dict(self.total_counts),
                "interval_counts": dict(self.interval_counts),
                "window_start": self.window_start,
                "history": list(self.history),
            }
