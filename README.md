<p align="center">
  <img src="assets/polusee_logo.png" alt="PoluSee — Pollution Tracker" width="180">
</p>

<h1 align="center">PoluSee Backend</h1>

<p align="center">
  <em>See the air, not just the traffic.</em>
</p>

<p align="center">
  Computer-vision backend for the <strong>PoluSee</strong> iOS app it watches public
  CCTV live streams, counts vehicles by class with YOLO, and serves the rolling
  traffic volume as JSON.
</p>

---

## Overview

PoluSee estimates street-level pollution from traffic volume. This repository is the
part that produces that volume: for every configured CCTV camera it opens the live
stream, tracks vehicles through a region of interest, classifies each one as **Bus**,
**Car**, **Motorcycle** or **Truck**, and rolls the counts up into a **7-minute
window**. Each closed window is appended to a SQLite table, and a small FastAPI
service exposes that table to the iOS client.

```
  YouTube live CCTV
         │
         │  yt-dlp resolves the direct HLS URL
         ▼
  ┌──────────────────┐   OpenCV capture
  │  CameraWorker    │   YOLO tracking (ByteTrack) + majority-vote classification
  │  (1 thread/cam)  │   7-minute roll-up
  └────────┬─────────┘
           │ INSERT
           ▼
   volume_history.db  ──────┬─────────────────────────────┐
     (SQLite, WAL)          │                             │
                            ▼                             ▼
                 ┌────────────────────┐        ┌────────────────────┐
                 │  Streamlit app     │        │  FastAPI server    │
                 │  live dashboard    │        │  JSON for iOS      │
                 │  localhost:8501    │        │  localhost:8000    │
                 └────────────────────┘        └─────────┬──────────┘
                                                         │  ngrok
                                                         ▼
                                                   PoluSee iOS app
```

> [!IMPORTANT]
> The camera workers live **inside the Streamlit process**. `server.py` is a read-only
> view over the database — it never opens a stream itself. To produce fresh data you
> must keep `streamlit_app.py` running; run `server.py` alongside it to publish that
> data to the app.

## Features

- **Multi-camera** — one background thread per camera, each with its own tracker state.
- **Live stream resolution** — `yt-dlp` turns a YouTube live URL into a direct HLS URL that OpenCV can read.
- **Class-stable counting** — an object is counted once, using a majority vote over its last frames, so flicker between classes does not inflate the totals. Fast vehicles that leave the frame before the vote threshold are still counted from their last observed class.
- **Region of interest** — detection is restricted to the middle 80% of the frame to skip noisy edges.
- **Auto-reconnect** — a dropped or stalled stream is retried automatically without losing accumulated counts.
- **Durable history** — every 7-minute window is written to SQLite in WAL mode, so the API can read while a worker writes.
- **Operator dashboard** — a themed Streamlit UI with annotated live frames, per-class totals, a countdown to the next roll-up, and the full history table.

## Project structure

| Path | Purpose |
| --- | --- |
| [`camera_worker.py`](camera_worker.py) | `CameraWorker` — capture, tracking, counting and 7-minute roll-up for one camera. |
| [`streamlit_app.py`](streamlit_app.py) | Live dashboard; also the process that starts and owns the workers. |
| [`server.py`](server.py) | FastAPI service consumed by the iOS app. |
| [`db.py`](db.py) | SQLite schema and queries for the `history` table. |
| [`cameras.py`](cameras.py) | Camera registry — name, live URL and street address. |
| [`deteksi_roi.py`](deteksi_roi.py) | Standalone webcam counter, used for testing the model locally. |
| [`.streamlit/config.toml`](.streamlit/config.toml) | PoluSee theme (palette, typography, radii) for the dashboard. |
| [`assets/`](assets/) | Logo lockup and pin mark used by the dashboard. |
| `model_custom.pt`, `best.pt` | YOLO weights — **not** in the repository, see below. |
| `volume_history.db` | Generated at runtime; ignored by git. |

## Requirements

- **Python 3.10+** (developed and tested on 3.12)
- **macOS** if you rely on the Safari-cookie path in `resolve_stream_url()` (see [Troubleshooting](#troubleshooting))
- Roughly 2 GB of disk for the dependencies, plus ~40 MB per model file

## Setup

```bash
git clone <repository-url>
cd PiBack

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Model weights

The YOLO weights are excluded from version control because they are ~40 MB each and
are replaced on every training run. Place them in the project root before starting:

| File | Used by | Notes |
| --- | --- | --- |
| `model_custom.pt` | `streamlit_app.py` → `CameraWorker` | The model the pipeline runs in production. |
| `best.pt` | `deteksi_roi.py` | Latest training checkpoint, used for local webcam testing. |

Both must be trained on the PoluSee 4-class set, in this exact order:

| Class ID | Label |
| --- | --- |
| `0` | Bus |
| `1` | Car |
| `2` | Motorcycle |
| `3` | Truck |

Point a different path at the pipeline by editing `MODEL_PATH` in
[`streamlit_app.py`](streamlit_app.py).

## Running

### 1. Dashboard and camera workers

```bash
streamlit run streamlit_app.py
```

Opens on <http://localhost:8501>. The workers start on first load and keep running
for the lifetime of the process. The first 7-minute window has to close before any
row appears — until then the history table shows a placeholder.

### 2. API server

In a second terminal, with the virtualenv active:

```bash
python server.py
```

Serves on <http://localhost:8000> — port 8000 rather than 80 so it needs no `sudo` on
macOS. Interactive docs are at `/docs`.

### 3. Exposing the API to the iOS app

The app needs a public URL, so tunnel the local server:

```bash
ngrok http --url=<your-ngrok-domain> 8000
```

Keep the tunnel port identical to the `uvicorn` port. Set the resulting HTTPS URL as
the API base URL in the iOS client.

### Local model check (optional)

```bash
python deteksi_roi.py
```

Runs `best.pt` against the default camera (device `0`) with an on-screen scoreboard.
Press `q` to quit.

## API reference

### `GET /`

The most recent closed window for every camera — this is what the iOS app polls.

```json
[
  {
    "kamera": "Simpang Gadong",
    "lokasi": "Jl. Raya Puncak - Cianjur No.57, Pandansari, Kec. Ciawi, Kabupaten Bogor, Jawa Barat 16720",
    "interval_menit": 7,
    "mulai": "2026-08-21 08:00:12",
    "selesai": "2026-08-21 08:07:12",
    "volume_kendaraan": {
      "motor": 214,
      "mobil": 96,
      "bus": 3,
      "truk": 11
    },
    "total": 324
  }
]
```

### `GET /history`

Past windows, newest first.

| Query parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `kamera` | string | *(all cameras)* | Filter by camera name, e.g. `Baranangsiang`. |
| `limit` | int | `100` | Maximum number of rows to return. |

```bash
curl "http://localhost:8000/history?kamera=Baranangsiang&limit=20"
```

The response objects use the same shape as `GET /`.

> [!NOTE]
> The JSON keys are Indonesian (`kamera`, `motor`, `truk`, …) because they are the
> contract the iOS client already consumes. The Python code is otherwise in English;
> `server.py` performs the mapping in `_format_entry()`.

## Data model

`volume_history.db`, table `history`:

| Column | Type | Description |
| --- | --- | --- |
| `id` | INTEGER PK | Autoincrement. |
| `kamera` | TEXT | Camera name, matching `cameras.py`. |
| `mulai` | TEXT | Window start, `YYYY-MM-DD HH:MM:SS`. |
| `selesai` | TEXT | Window end, same format. |
| `bus`, `mobil`, `motor`, `truk` | INTEGER | Vehicles counted in the window, per class. |
| `total` | INTEGER | Sum of the four class columns. |

The database opens in **WAL** mode so the FastAPI reader is never blocked by a worker
write; writes themselves are serialised through a process-level lock in `db.py`.

## Configuration

| Setting | Location | Default |
| --- | --- | --- |
| Camera list (name, URL, address) | `cameras.py` → `CAMERAS` | Simpang Gadong, Baranangsiang |
| Roll-up interval | `camera_worker.py` → `INTERVAL_SECONDS` | `7 * 60` |
| Detection confidence | `camera_worker.py` → `model.track(conf=...)` | `0.40` |
| Frames required for a class vote | `camera_worker.py` → `MIN_FRAME_VOTING` | `4` |
| Reconnect delay | `camera_worker.py` → `RECONNECT_DELAY_SECONDS` | `5` |
| Region of interest margin | `CameraWorker(roi_margin=...)` | `0.1` (middle 80%) |
| Model path | `streamlit_app.py` → `MODEL_PATH` | `model_custom.pt` |
| Dashboard refresh rate | `streamlit_app.py` → `REFRESH_SECONDS` | `2` |

Changing `INTERVAL_SECONDS` also means updating the hardcoded `"interval_menit": 7`
in `server.py`, otherwise the API will advertise the wrong window length.

To add a camera, append an entry to `CAMERAS` and restart Streamlit — a worker is
created per entry automatically.

## Troubleshooting

**`Sign in to confirm you're not a bot` from yt-dlp**
`resolve_stream_url()` reads cookies from Safari (`cookiesfrombrowser: ("safari",)`),
so you need to be signed in to YouTube in Safari on the machine running the worker.
On another OS or browser, change that option to your browser, e.g. `("chrome",)`.

**`Cannot open stream` / constant reconnecting**
The live URL may have ended or gone private — YouTube live URLs are not permanent.
Verify it in a browser and update `cameras.py`.

**`Failed to load model`**
The weights are missing from the project root. See [Model weights](#model-weights).

**Tracking errors mentioning ByteTrack or `lap`**
The `lap` solver is required by the tracker; reinstall with
`pip install --force-reinstall lap`.

**The history table stays empty**
That is expected for the first 7 minutes after startup — a row only exists once a
window closes. Confirm the camera badge reads **Live** on the dashboard.

**No new data reaches the iOS app**
`server.py` alone produces nothing. Check that the Streamlit process is still running.

## License

Proprietary part of the Pirless team project. All rights reserved.
