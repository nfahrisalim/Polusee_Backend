from typing import Optional

from fastapi import FastAPI
import uvicorn

import db
from cameras import LOKASI_PER_KAMERA

app = FastAPI()


@app.on_event("startup")
def on_startup():
    db.init_db()


def _format_entry(row):
    return {
        "kamera": row["kamera"],
        "lokasi": LOKASI_PER_KAMERA.get(row["kamera"]),
        "interval_menit": 7,
        "mulai": row["mulai"],
        "selesai": row["selesai"],
        "volume_kendaraan": {
            "motor": row["motor"],
            "mobil": row["mobil"],
            "bus": row["bus"],
            "truk": row["truk"],
        },
        "total": row["total"],
    }


# Endpoint ini yang akan diakses oleh URL Ngrok secara global
# Mengembalikan rekap volume kendaraan 7 menit terakhir untuk tiap kamera,
# diambil dari volume_history.db yang ditulis oleh CameraWorker (camera_worker.py)
@app.get("/")
def get_volume_kendaraan():
    rows = db.get_latest_per_kamera()
    return [_format_entry(r) for r in rows]


@app.get("/history")
def get_history(kamera: Optional[str] = None, limit: int = 100):
    rows = db.get_history(kamera=kamera, limit=limit)
    return [_format_entry(r) for r in rows]


if __name__ == "__main__":
    # Port 8000 dipilih agar tidak butuh sudo di macOS.
    # Samakan port ini dengan port tujuan di command ngrok, mis:
    # ngrok http --url=<domain-ngrok-kamu> 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)