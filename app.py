import tempfile
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from alerts import SmsAlerter, StockMonitor
from config import COCO_MODEL, DEFAULT_CONFIDENCE, DEFAULT_THRESHOLD, DEFAULT_TRACKED, MODEL_PATH, SMOOTHING_WINDOW
import detector as detector_mod
import importlib
importlib.reload(detector_mod)
from detector import CountSmoother, CumulativeCounter, Detector, display

st.set_page_config(page_title="ShelfWatch", page_icon="🥕", layout="wide")

STATUS_COLORS = {"Out of stock": "#B3261E", "Low": "#A15C00", "In stock": "#2F7D4A"}


@st.cache_resource
def fetch_detector(path, coco_path):
    return Detector(path, coco_path)


@st.cache_resource
def load_alerter():
    try:
        cfg = dict(st.secrets.get("twilio", {}))
    except Exception:
        cfg = {}
    return SmsAlerter(
        cfg.get("sid"),
        cfg.get("auth_token"),
        cfg.get("from_number"),
        cfg.get("to_number"),
        cfg.get("messaging_service_sid"),
    )


def to_rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def decode_image(file):
    return cv2.imdecode(np.frombuffer(file.getvalue(), np.uint8), cv2.IMREAD_COLOR)


def open_capture(source):
    return cv2.VideoCapture(int(source) if source.strip().isdigit() else source.strip())


def style_status(value):
    return f"color: {STATUS_COLORS.get(value, 'inherit')}; font-weight: 600"


def render_stock(slot, counts, source):
    if people:
        return
    active_tracked = tracked if tracked else list(set(counts.keys()).union(monitor.alerted))
    if not active_tracked:
        slot.info("Choose items to track in the sidebar or wait for items to be detected.")
        return
    rows = monitor.evaluate(counts, active_tracked, threshold)
    if sms_on:
        monitor.alert(counts, active_tracked, threshold, alerter, source)
    table = pd.DataFrame(rows).style.map(style_status, subset=["Status"])
    slot.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={"Reorder": st.column_config.LinkColumn("Reorder", display_text="Order now")},
    )


def render_counts(slot, counts, column="Count"):
    if not counts:
        slot.info("Nothing detected yet. Try lowering the confidence in the sidebar.")
        return
    rows = [{"Item": display(item), column: n} for item, n in counts.most_common()]
    slot.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def count_slots():
    slots = {}
    if not people:
        st.subheader("Stock levels")
        slots["stock"] = st.empty()
    live_col, total_col = st.columns(2)
    live_col.subheader("Live count")
    live_col.caption("In view right now")
    slots["live"] = live_col.empty()
    total_col.subheader("Cumulative count")
    total_col.caption("Unique objects seen so far")
    slots["total"] = total_col.empty()
    return slots


def render_counts_panel(slots, live, total, source):
    if "stock" in slots:
        render_stock(slots["stock"], live, source)
    render_counts(slots["live"], live)
    render_counts(slots["total"], total, "Total seen")


def render_results(frame, source, imgsz=None):
    detections = detector.detect(frame, confidence, people, imgsz=imgsz)
    counts = detector.count(detections)
    left, right = st.columns([3, 2], gap="large")
    left.image(to_rgb(detector.annotate(frame, detections)), width="stretch")
    with right:
        st.metric(f"{noun} detected", len(detections))
        if not people:
            st.subheader("Stock levels")
            render_stock(st.empty(), counts, source)
        st.subheader("Count")
        render_counts(st.empty(), counts)


if "monitor" not in st.session_state:
    st.session_state.monitor = StockMonitor()
monitor = st.session_state.monitor
alerter = load_alerter()

with st.sidebar:
    st.header("Settings")
    target = st.segmented_control("Detect", ["Objects", "People"], default="Objects", width="stretch") or "Objects"
    people = target == "People"
    noun = "People" if people else "Items"
    use_coco = people or st.toggle(
        "Include everyday objects",
        value=True,
        help="Groceries are always detected. Turn this on to also find about 70 everyday things like bottles, cups, bowls and phones.",
    )

    if not MODEL_PATH.exists():
        st.error(f"No model file at {MODEL_PATH}. Put best.pt next to app.py.")
        st.stop()
    with st.spinner("Loading models"):
        detector = fetch_detector(str(MODEL_PATH), COCO_MODEL if use_coco else None)
    catalog = detector.catalog(people)

    confidence = st.slider("Confidence", 0.10, 0.90, DEFAULT_CONFIDENCE, 0.05)
    if people:
        threshold, tracked = DEFAULT_THRESHOLD, []
    else:
        threshold = st.number_input("Low stock at or below", min_value=0, max_value=100, value=DEFAULT_THRESHOLD)
        tracked = st.multiselect(
            "Items to track",
            catalog,
            default=[item for item in DEFAULT_TRACKED if item in catalog],
            format_func=display,
        )

    st.divider()
    sms_on = st.toggle("Send SMS alerts", value=False)
    if alerter.enabled:
        st.caption("Twilio is configured.")
    else:
        st.caption("Twilio isn't configured, so alerts are logged but not sent. Add keys to .streamlit/secrets.toml.")
    if st.button("Reset alerts", width="stretch"):
        monitor.reset()
        st.toast("Alerts reset. Low items will alert again.")

st.title("ShelfWatch")
st.caption("Count groceries from a photo, a video, or a live camera, and get a text when something runs low.")

photo_tab, video_tab, live_tab, log_tab = st.tabs(["Photo", "Video", "Live camera", "Alert log"])

with photo_tab:
    mode = st.radio("Source", ["Upload a photo", "Take a photo"], horizontal=True, label_visibility="collapsed")
    image_file = (
        st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png", "webp"])
        if mode == "Upload a photo"
        else st.camera_input("Take a photo")
    )
    if image_file:
        frame = decode_image(image_file)
        if frame is None:
            st.error("That file couldn't be read as an image. Try a JPG or PNG.")
        else:
            render_results(frame, "Photo", imgsz=1280)

with video_tab:
    video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    stride = st.slider("Analyse every Nth frame", 1, 15, 5, help="Higher is faster but less thorough.")
    if video_file and st.button("Analyse video", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(video_file.name).suffix) as tmp:
            tmp.write(video_file.getvalue())
            input_path = tmp.name
        cap = cv2.VideoCapture(input_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_path = str(Path(tempfile.gettempdir()) / "shelfwatch_output.mp4")
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), max(fps / stride, 1), (width, height))

        progress = st.progress(0.0, text="Analysing video")
        preview = st.empty()
        smoother = CountSmoother(SMOOTHING_WINDOW)
        cumulative = CumulativeCounter(detector, confidence, fps / stride)
        smoothed = Counter()
        total = Counter()
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % stride == 0:
                detections = detector.detect(frame, confidence, people)
                smoothed = smoother.update(detector.count(detections))
                total = cumulative.update(detections)
                annotated = detector.annotate(frame, detections)
                writer.write(annotated)
                preview.image(to_rgb(annotated), width="stretch")
            index += 1
            progress.progress(min(index / total, 1.0), text=f"Analysing video: frame {index} of {total}")
        cap.release()
        writer.release()
        Path(input_path).unlink(missing_ok=True)
        progress.empty()

        if index == 0:
            st.error("No frames could be read from this video. Try re-exporting it as MP4.")
        else:
            if not people:
                st.subheader("Stock at the end of the video")
                render_stock(st.empty(), smoothed, "Video")
            left, right = st.columns(2, gap="large")
            with left:
                st.subheader("Live count")
                st.caption("In view at the end of the video")
                render_counts(st.empty(), smoothed)
            with right:
                st.subheader("Cumulative count")
                st.caption("Unique objects seen across the whole video")
                render_counts(st.empty(), total, "Total seen")
            st.download_button(
                "Download annotated video",
                Path(output_path).read_bytes(),
                file_name="shelfwatch_output.mp4",
                mime="video/mp4",
            )

with live_tab:
    source = st.text_input("Camera", "0", help="Use 0 for this computer's webcam, or paste an RTSP or HTTP stream URL.")
    running = st.toggle("Start live detection", help="The cumulative count starts from zero each time you turn this on.")
    if running:
        cap = open_capture(source)
        if not cap.isOpened():
            st.error(f"Couldn't open camera {source}. Check the index or URL and that nothing else is using it.")
        else:
            frame_slot, side = st.columns([3, 2], gap="large")
            frame_view = frame_slot.empty()
            with side:
                fps_view = st.empty()
                slots = count_slots()
            smoother = CountSmoother(SMOOTHING_WINDOW)
            cumulative = CumulativeCounter(detector, confidence, cap.get(cv2.CAP_PROP_FPS) or 30)
            last_table = 0.0
            try:
                while True:
                    started = time.time()
                    ok, frame = cap.read()
                    if not ok:
                        st.warning("The camera stopped sending frames.")
                        break
                    detections = detector.detect(frame, confidence, people)
                    smoothed = smoother.update(detector.count(detections))
                    total = cumulative.update(detections)
                    frame_view.image(to_rgb(detector.annotate(frame, detections)), width="stretch")
                    if time.time() - last_table > 1:
                        render_counts_panel(slots, smoothed, total, f"Camera {source}")
                        last_table = time.time()
                    fps_view.caption(f"{1 / max(time.time() - started, 1e-6):.1f} frames per second")
            finally:
                cap.release()

with log_tab:
    if monitor.log:
        st.dataframe(pd.DataFrame(monitor.log), hide_index=True, width="stretch")
    else:
        st.info("No alerts yet. Turn on SMS alerts in the sidebar and low items will show up here.")
