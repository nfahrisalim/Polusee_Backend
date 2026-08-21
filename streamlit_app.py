import base64
import html
import time
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from camera_worker import CameraWorker, CLASS_NAMES, CLASSES, INTERVAL_SECONDS
from cameras import CAMERAS

MODEL_PATH = "model_custom.pt"

REFRESH_SECONDS = 2

ALL_CAMERAS = "All cameras"

# The PoluSee pin, flattened onto white (the source PNG is transparent). Just the pin,
# not the full lockup: the wordmark is already the page heading, and it would be
# unreadable at favicon size anyway. assets/polusee_logo.png holds the full lockup.
MARK_PATH = "assets/polusee_mark.png"

# (css class, label) for the status badge on each camera
STATUS_PILL = {
    "starting": ("wait", "Starting"),
    "connecting": ("wait", "Connecting"),
    "running": ("live", "Live"),
    "reconnecting": ("err", "Reconnecting"),
    "error": ("err", "Error"),
}

st.set_page_config(page_title="CCTV Traffic Volume Monitoring", layout="wide", page_icon=MARK_PATH)


CSS = """
<style>
:root {
  --teal: #0FA37F;
  --mint: #7FF8CF;
  --sky: #5BC0EB;
  --pale: #BFE9FF;
  --shell: #FAF8F5;
  /* ink sits outside the 5-colour palette and is used for text only: all five
     palette colours are too light to read as body text on #FAF8F5 */
  --ink: #0B3A44;
  --sf: "SF Pro Display", "SF Pro Text", "SF Pro", -apple-system, BlinkMacSystemFont,
        "Helvetica Neue", Arial, sans-serif;
}

.stApp, .stApp button, .stApp input, .stApp select, .stApp textarea, .stApp label,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp span, .stApp div {
  font-family: var(--sf);
}
.stApp code, .stApp pre, .stApp kbd {
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
}

[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1100px 500px at 12% -8%, rgba(127, 248, 207, .38), transparent 60%),
    radial-gradient(900px 460px at 88% -12%, rgba(191, 233, 255, .55), transparent 60%),
    var(--shell);
}
[data-testid="stAppHeader"] { background: transparent; }

/* use the full window width so the camera frames stay large */
[data-testid="stMainBlockContainer"] {
  max-width: 100%;
  padding: 2.2rem 2.4rem 3rem;
}

/* ---------- page header ---------- */
.page-head { display: flex; gap: 1rem; align-items: center; margin: .2rem 0 1.5rem; }
.page-badge {
  width: 64px; height: 64px; border-radius: 18px; flex: none;
  display: grid; place-items: center; overflow: hidden;
  background: #FFFFFF;
  border: 1px solid var(--pale);
  box-shadow: 0 12px 26px -14px rgba(11, 58, 68, .5);
}
.page-badge img { width: 100%; height: 100%; object-fit: contain; display: block; }
.page-head h1 {
  margin: 0; font-size: 1.9rem; font-weight: 700;
  letter-spacing: -.02em; color: var(--ink);
}
.page-head p {
  margin: .32rem 0 0; font-size: .93rem; max-width: 74ch;
  color: var(--ink); opacity: .68;
}

/* ---------- section label ---------- */
.section-label {
  display: flex; align-items: center; gap: .7rem;
  font-size: .76rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase;
  color: var(--teal); margin: 1.4rem 0 .7rem;
}
.section-label::after {
  content: ""; flex: 1; height: 1px;
  background: linear-gradient(90deg, var(--pale), transparent);
}

/* ---------- camera ---------- */
.cam-head {
  display: flex; align-items: center; gap: .5rem;
  flex-wrap: wrap; margin: .15rem 0 .6rem;
}
.cam-name {
  margin-right: auto; font-size: 1.2rem; font-weight: 700;
  letter-spacing: -.01em; color: var(--ink);
}
.cam-pill {
  display: inline-flex; align-items: center; gap: .45rem;
  padding: .28rem .72rem; border-radius: 999px;
  font-size: .78rem; font-weight: 600; white-space: nowrap;
}
.cam-pill .dot { width: .48rem; height: .48rem; border-radius: 50%; background: currentColor; }
.cam-pill.live { background: var(--mint); color: #07564A; }
.cam-pill.wait { background: var(--pale); color: var(--ink); }
.cam-pill.err  { background: #FFE0DC; color: #8E2F26; }
.cam-pill.live .dot { animation: blip 1.6s ease-in-out infinite; }
@keyframes blip {
  0%, 100% { opacity: 1; transform: scale(1); }
  50%      { opacity: .3; transform: scale(.75); }
}

.st-key-camera-zone [data-testid="stImage"] img {
  display: block; width: 100%;
  border-radius: 18px;
  border: 1px solid var(--pale);
  box-shadow: 0 22px 44px -26px rgba(11, 58, 68, .55);
}

/* keep Streamlit's native fullscreen (expand) control visible instead of hover-only */
[data-testid="stFullScreenFrame"] [data-testid="stElementToolbar"] {
  opacity: 1;
  background: rgba(255, 255, 255, .93);
  border: 1px solid var(--pale);
  border-radius: 10px;
}

.st-key-camera-zone [data-testid="stMetric"] {
  background: linear-gradient(165deg, var(--shell), rgba(191, 233, 255, .55));
  border: 1px solid var(--pale);
  border-radius: 14px;
  padding: .62rem .7rem;
}
.st-key-camera-zone [data-testid="stMetricValue"] { color: var(--teal); font-weight: 700; }
.st-key-camera-zone [data-testid="stMetricLabel"] p {
  color: var(--ink); opacity: .72; font-weight: 600; font-size: .82rem;
}

/* ---------- view switcher ---------- */
[data-testid="stButtonGroup"] { margin-bottom: .2rem; }

/* ---------- history ---------- */
.st-key-history-zone [data-testid="stDataFrame"] {
  border-radius: 14px; overflow: hidden; border: 1px solid var(--pale);
}
</style>
"""


@st.cache_resource(show_spinner=False)
def get_workers():
    workers = []
    for cam in CAMERAS:
        worker = CameraWorker(cam["name"], cam["url"], MODEL_PATH)
        worker.start()
        workers.append(worker)
    return workers


@st.cache_data(show_spinner=False)
def image_data_uri(path):
    """Inline the image so the markdown header needs no static file serving."""
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def render_page_head():
    st.markdown(
        f"""
        <div class="page-head">
          <div class="page-badge"><img src="{image_data_uri(MARK_PATH)}" alt="PoluSee"></div>
          <div>
            <h1>Polusee</h1>
            <p>See the air not just the traffic.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_slots(workers):
    """Create the per-camera placeholders once, so the refresh loop only swaps content."""
    if len(workers) == 1:
        holders = [st.container()]
    else:
        holders = st.columns(len(workers), gap="large")

    slots = []
    for holder in holders:
        with holder:
            slots.append(
                {
                    "head": st.empty(),
                    "frame": st.empty(),
                    "error": st.empty(),
                    "metrics": st.empty(),
                    "progress": st.empty(),
                }
            )
    return slots


def refresh_camera(worker, slot):
    snap = worker.snapshot()

    pill_cls, pill_text = STATUS_PILL.get(snap["status"], ("wait", snap["status"]))
    slot["head"].markdown(
        f'<div class="cam-head">'
        f'<span class="cam-name">{html.escape(worker.name)}</span>'
        f'<span class="cam-pill {pill_cls}"><span class="dot"></span>{html.escape(pill_text)}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )

    if snap["frame"] is not None:
        frame_rgb = cv2.cvtColor(snap["frame"], cv2.COLOR_BGR2RGB)
        slot["frame"].image(frame_rgb, width="stretch")
    else:
        slot["frame"].info("Connecting to stream...")

    if snap["status"] in ("reconnecting", "error") and snap["error"]:
        slot["error"].caption(snap["error"])
    else:
        slot["error"].empty()

    with slot["metrics"].container():
        metric_cols = st.columns(len(CLASSES))
        for i, c in enumerate(CLASSES):
            metric_cols[i].metric(CLASS_NAMES[c], snap["total_counts"][c])

    elapsed = time.time() - snap["window_start"]
    remaining = max(0, INTERVAL_SECONDS - elapsed)
    mm, ss = divmod(int(remaining), 60)
    slot["progress"].progress(
        min(1.0, elapsed / INTERVAL_SECONDS),
        text=f"Next 7-minute roll-up in {mm:02d}:{ss:02d}",
    )


def render_history(workers, placeholder):
    rows = []
    for worker in workers:
        rows.extend(worker.snapshot()["history"])

    with placeholder.container():
        if rows:
            df = pd.DataFrame(rows).sort_values("End", ascending=False)
            st.dataframe(df, width="stretch", hide_index=True)
        else:
            st.info("No data yet — waiting for the first 7-minute interval of each camera to finish.")


def main():
    st.markdown(CSS, unsafe_allow_html=True)
    render_page_head()

    workers = get_workers()
    names = [w.name for w in workers]

    # Built once outside the refresh loop and given a stable key: a widget recreated
    # on every loop pass would get a fresh identity each time and never see its click.
    view = st.segmented_control(
        "View",
        [ALL_CAMERAS] + names,
        default=ALL_CAMERAS,
        key="view_mode",
        label_visibility="collapsed",
    )
    if view is None:
        view = ALL_CAMERAS
    st.caption(
        "Pick a single camera to view it full width, or use the ⛶ button on a frame "
        "to expand it to true fullscreen."
    )

    shown = workers if view == ALL_CAMERAS else [w for w in workers if w.name == view]

    camera_zone = st.container(key="camera-zone")
    with camera_zone:
        st.markdown('<div class="section-label">Live monitoring</div>', unsafe_allow_html=True)
        slots = build_slots(shown)

    history_zone = st.container(key="history-zone")
    with history_zone:
        st.markdown('<div class="section-label">Volume history — every 7 minutes</div>', unsafe_allow_html=True)
        history_ph = st.empty()

    while True:
        for worker, slot in zip(shown, slots):
            refresh_camera(worker, slot)
        render_history(workers, history_ph)
        time.sleep(REFRESH_SECONDS)


if __name__ == "__main__":
    main()
