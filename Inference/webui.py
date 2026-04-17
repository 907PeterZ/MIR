import os
import shutil
import gc
import atexit
import signal
import json
import html
import pickle
import subprocess
import time
import ctypes
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import whisper
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.io import wavfile

from extract_all_features import (
    clear_feature_runtime_caches,
    load_fasterrcnn_model,
    load_wav2vec_feature_bundle,
    run_detect_speaker,
    run_video_feature_extraction,
    run_audio_feature_extraction_from_wav,
)
from inference import (
    run_inference,
    INTENT_LABELS,
    load_inference_model,
    clear_inference_model_cache,
)


ROOT = os.path.dirname(os.path.abspath(__file__))
INPUT_VIDEO_PATH = os.path.join(ROOT, "raw_video", "single_input.mp4")
PREPROCESSED_ROOT = os.path.join(ROOT, "preprocessed_output")
PERSON_DETECTOR_NAME = "FasterRCNN"
FACE_DETECTOR_NAME = "YuNet"

st.set_page_config(page_title="Multimodal Inference Web Demo", layout="wide")


@st.cache_resource(show_spinner=True)
def load_shared_detector_model():
    return load_fasterrcnn_model()


@st.cache_resource(show_spinner=True)
def load_shared_inference_bundle():
    return load_inference_model()


@st.cache_resource(show_spinner=True)
def load_shared_audio_feature_bundle():
    return load_wav2vec_feature_bundle(device="cpu")


@st.cache_resource(show_spinner=True)
def load_shared_whisper_tiny_model():
    return whisper.load_model("tiny", device="cpu")


def extract_text(wav_path, model_size="tiny", device=None, shared_model=None):
    if not os.path.exists(wav_path):
        raise FileNotFoundError("Missing wav file: {}".format(wav_path))
    owns_model = shared_model is None
    if owns_model:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model(model_size, device=device)
    else:
        model = shared_model
    try:
        result = model.transcribe(wav_path)
        return result["text"].strip()
    finally:
        if owns_model:
            # Release Whisper after use when not using shared resident model.
            try:
                del model
            except Exception:
                pass
            torch_cleanup()


def save_uploaded_video(uploaded_file):
    os.makedirs(os.path.dirname(INPUT_VIDEO_PATH), exist_ok=True)
    with open(INPUT_VIDEO_PATH, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return INPUT_VIDEO_PATH


def _get_detect_root():
    shm_root = "/dev/shm"
    if os.path.isdir(shm_root) and os.access(shm_root, os.W_OK):
        return os.path.join(shm_root, "mintrec_detect_speaker")
    return os.path.join(PREPROCESSED_ROOT, "detect_speaker")


def _get_reset_cleanup_paths():
    paths = [os.path.dirname(INPUT_VIDEO_PATH), PREPROCESSED_ROOT]
    # Always include both potential detect roots to avoid RAM-disk residue.
    paths.append(os.path.join(PREPROCESSED_ROOT, "detect_speaker"))
    paths.append(os.path.join("/dev/shm", "mintrec_detect_speaker"))
    # De-duplicate while preserving order.
    dedup_paths = []
    seen = set()
    for p in paths:
        if p not in seen:
            dedup_paths.append(p)
            seen.add(p)
    return dedup_paths


def create_speaker_overlay_video(video_path, best_persons_path, output_path):
    if not os.path.exists(best_persons_path):
        raise FileNotFoundError("Missing best_persons.npy: {}".format(best_persons_path))

    bboxes = np.load(best_persons_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open video: {}".format(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w = width if width % 2 == 0 else width - 1
    out_h = height if height % 2 == 0 else height - 1
    if out_w <= 0:
        out_w = width
    if out_h <= 0:
        out_h = height

    base, ext = os.path.splitext(output_path)
    if not ext:
        ext = ".mp4"
    raw_overlay_path = base + "_raw" + ext
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(raw_overlay_path, fourcc, fps, (out_w, out_h))

    try:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame_idx < len(bboxes):
                x1, y1, x2, y2 = [int(v) for v in bboxes[frame_idx].tolist()]
                if not (x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0):
                    x1 = max(0, min(width - 1, x1))
                    y1 = max(0, min(height - 1, y1))
                    x2 = max(0, min(width - 1, x2))
                    y2 = max(0, min(height - 1, y2))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 220, 20), 2)
                    cv2.putText(
                        frame,
                        "Active Speaker ROI",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (20, 220, 20),
                        2,
                        cv2.LINE_AA,
                    )
            if frame.shape[1] != out_w or frame.shape[0] != out_h:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    if not os.path.exists(raw_overlay_path) or os.path.getsize(raw_overlay_path) == 0:
        raise RuntimeError("Failed to generate overlay video at {}".format(raw_overlay_path))

    base_output, _ = os.path.splitext(output_path)
    mp4_output_path = base_output + ".mp4"
    webm_output_path = base_output + ".webm"
    transcode_profiles = [
        (mp4_output_path, ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-c:a", "aac", "-b:a", "128k"]),
        (mp4_output_path, ["-c:v", "libopenh264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-c:a", "aac", "-b:a", "128k"]),
        (webm_output_path, ["-c:v", "libvpx-vp9", "-crf", "32", "-b:v", "0", "-c:a", "libopus", "-b:a", "96k"]),
        (webm_output_path, ["-c:v", "libvpx", "-crf", "10", "-b:v", "1M", "-c:a", "libvorbis", "-b:a", "96k"]),
        (mp4_output_path, ["-c:v", "mpeg4", "-q:v", "2", "-movflags", "+faststart", "-c:a", "aac", "-b:a", "128k"]),
    ]
    for target_path, profile in transcode_profiles:
        if os.path.exists(target_path):
            os.remove(target_path)
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            raw_overlay_path,
            "-i",
            video_path,
            "-map",
            "0:v:0",
            "-map",
            "1:a:0?",
        ] + profile + [
            "-shortest",
            target_path,
            "-loglevel",
            "error",
        ]
        proc = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode == 0 and os.path.exists(target_path) and os.path.getsize(target_path) > 0:
            return target_path

    # Keep and return raw local overlay video when all transcode profiles fail.
    return raw_overlay_path


def plot_waveform(wav_path):
    sr, y = wavfile.read(wav_path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))

    target_points = 4500
    if len(y) > target_points:
        stride = max(1, len(y) // target_points)
        y_plot = y[::stride]
        t_plot = (np.arange(len(y_plot)) * stride) / float(sr)
    else:
        y_plot = y
        t_plot = np.arange(len(y_plot)) / float(sr)

    fig, ax = plt.subplots(figsize=(5.8, 1.9), dpi=110)
    ax.plot(t_plot, y_plot, linewidth=0.75, color="#1565c0")
    ax.fill_between(t_plot, y_plot, 0, color="#64b5f6", alpha=0.25)
    ax.set_title("Audio Waveform", fontsize=11)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Amp", fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_probabilities(infer_result, top_k=8):
    probs = np.array(infer_result["probs"], dtype=np.float32)
    pred_label = infer_result["intent"]
    order = np.argsort(probs)[::-1][:top_k]
    sorted_probs = probs[order]
    sorted_labels = [INTENT_LABELS[i] for i in order]
    colors = ["#1d5fbf"] * len(sorted_labels)
    if pred_label in sorted_labels:
        colors[sorted_labels.index(pred_label)] = "#2e7d32"

    fig_h = max(2.15, 0.24 * len(sorted_labels) + 0.78)
    fig, ax = plt.subplots(figsize=(6.0, fig_h), dpi=115)
    y_pos = np.arange(len(sorted_labels))
    ax.barh(y_pos, sorted_probs, color=colors, height=0.58)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Probability", fontsize=9)
    ax.set_title("Intent Class Probabilities (Top-10)", fontsize=11)
    ax.grid(axis="x", alpha=0.22)
    fig.tight_layout()
    return fig


def render_text_box(text):
    safe_text = html.escape(text if text else "(empty)")
    return "<div class='text-box'><strong>Text:</strong><br>{}</div>".format(safe_text)


def render_result_banner(placeholder, infer_result):
    intent = html.escape(str(infer_result.get("intent", "-")))
    binary = html.escape(str(infer_result.get("binary", "-")))
    placeholder.markdown(
        """
<div class="result-banner">
  <span class="result-tag">Final Result</span>
  <span class="result-main">Intent: {}</span>
  <span class="result-sep">|</span>
  <span class="result-main">Binary: {}</span>
</div>
""".format(intent, binary),
        unsafe_allow_html=True,
    )


def _safe_video_duration_seconds(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps and fps > 0 and frame_count and frame_count > 0:
            return float(frame_count / fps)
    finally:
        cap.release()
    return 0.0


def _load_detect_stage_timing(detect_path):
    timing_path = os.path.join(detect_path, "pywork", "stage_timing.json")
    if not os.path.exists(timing_path):
        return {}
    try:
        with open(timing_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _fmt_seconds(value):
    return "{:.2f} s".format(float(value))


def render_timing_table(placeholder, timing_stats):
    table_html = """
<div class="timing-wrap">
  <table class="timing-table">
    <thead>
      <tr>
        <th>视频时长</th>
        <th>特征提取时长</th>
        <th>MULT推理时长</th>
        <th>端到端时长</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>{}</td>
        <td>{}</td>
        <td>{}</td>
        <td>{}</td>
      </tr>
    </tbody>
  </table>
</div>
""".format(
        _fmt_seconds(timing_stats.get("video_duration_s", 0.0)),
        _fmt_seconds(timing_stats.get("feature_extraction_s", 0.0)),
        _fmt_seconds(timing_stats.get("mult_inference_s", 0.0)),
        _fmt_seconds(timing_stats.get("end_to_end_s", 0.0)),
    )
    placeholder.markdown(table_html, unsafe_allow_html=True)


def run_pipeline_with_live_visuals(
    video_path,
    detector_model,
    inference_bundle,
    audio_feature_bundle,
    whisper_model,
    ui,
):
    total_start = time.perf_counter()
    video_duration_s = _safe_video_duration_seconds(video_path)
    feature_start = time.perf_counter()

    if os.path.exists(PREPROCESSED_ROOT):
        shutil.rmtree(PREPROCESSED_ROOT, ignore_errors=True)
    os.makedirs(PREPROCESSED_ROOT, exist_ok=True)

    detect_root = _get_detect_root()
    os.makedirs(detect_root, exist_ok=True)

    try:
        # Keep GPU focused on visual/TalkNet path on smaller cards.
        aux_modal_device = "cpu"

        def _transcribe_from_wav(wav_path_local, transcribe_device):
            text_local = extract_text(
                wav_path_local,
                model_size="tiny",
                device=transcribe_device,
                shared_model=whisper_model,
            )
            return text_local, wav_path_local

        video_feats_path = None
        audio_feats_path = None
        wav_path = None
        text = ""
        detect_stage_timing = {}
        overlay_path = ""
        detect_path = ""
        best_persons_path = ""

        with ThreadPoolExecutor(max_workers=3) as executor:
            ui["status"].info(
                "Step 1/3: TalkNet active speaker detection (person: {}, face: {})".format(
                    PERSON_DETECTOR_NAME,
                    FACE_DETECTOR_NAME,
                )
            )
            detect_root, detect_path = run_detect_speaker(
                video_path,
                detect_root,
                detector_model=detector_model,
            )
            detect_stage_timing = _load_detect_stage_timing(detect_path)
            if detect_stage_timing:
                person_s = float(detect_stage_timing.get("person_detect_s", 0.0))
                face_s = float(detect_stage_timing.get("face_detect_s", 0.0))
                track_s = float(detect_stage_timing.get("face_track_s", 0.0))
                ui["video_meta"].caption(
                    "TalkNet timing: person {:.2f}s | face {:.2f}s | track {:.2f}s".format(
                        person_s,
                        face_s,
                        track_s,
                    )
                )

            best_persons_path = os.path.join(detect_path, "pywork", "best_persons.npy")
            talknet_wav_path = os.path.join(detect_path, "pyavi", "audio.wav")
            if not os.path.exists(talknet_wav_path):
                raise FileNotFoundError("Missing TalkNet wav: {}".format(talknet_wav_path))
            overlay_path = os.path.join(PREPROCESSED_ROOT, "speaker_overlay.mp4")
            overlay_path = create_speaker_overlay_video(video_path, best_persons_path, overlay_path)
            ui["video"].video(overlay_path)

            ui["status"].info("Step 2/3: Visual/Audio/Text feature extraction (parallel)")
            visual_future = executor.submit(
                run_video_feature_extraction,
                PREPROCESSED_ROOT,
                detect_root,
                detector_model,
            )
            audio_future = executor.submit(
                run_audio_feature_extraction_from_wav,
                talknet_wav_path,
                PREPROCESSED_ROOT,
                aux_modal_device,
                audio_feature_bundle,
                True,
            )
            text_future = executor.submit(_transcribe_from_wav, talknet_wav_path, aux_modal_device)
            futures = {
                visual_future: "visual",
                audio_future: "audio",
                text_future: "text",
            }

            for future in as_completed(futures):
                task = futures[future]
                if task == "visual":
                    video_feats_path = future.result()
                    with open(video_feats_path, "rb") as f:
                        feat_data = pickle.load(f)
                    feat_shape = np.asarray(feat_data).shape
                    ui["video_meta"].caption("Video feature shape: {}".format(feat_shape))
                elif task == "audio":
                    audio_feats_path, wav_path = future.result()
                    wave_fig = plot_waveform(wav_path)
                    ui["audio"].pyplot(wave_fig, use_container_width=True)
                    plt.close(wave_fig)
                elif task == "text":
                    text, text_wav_path = future.result()
                    if wav_path is None:
                        wav_path = text_wav_path
                    ui["text"].markdown(render_text_box(text), unsafe_allow_html=True)

        if video_feats_path is None or audio_feats_path is None:
            raise RuntimeError("Parallel feature extraction failed: missing video/audio features.")

        feature_extraction_s = time.perf_counter() - feature_start
        ui["status"].info("Step 3/3: MULT inference")
        infer_start = time.perf_counter()
        infer_result = run_inference(
            text_input=text,
            video_pkl=video_feats_path,
            audio_pkl=audio_feats_path,
            model_bundle=inference_bundle,
        )
        mult_inference_s = time.perf_counter() - infer_start
        end_to_end_s = time.perf_counter() - total_start

        timing_stats = {
            "video_duration_s": float(video_duration_s),
            "feature_extraction_s": float(feature_extraction_s),
            "mult_inference_s": float(mult_inference_s),
            "end_to_end_s": float(end_to_end_s),
            "talknet_person_detect_s": float(detect_stage_timing.get("person_detect_s", 0.0)),
            "talknet_face_detect_s": float(detect_stage_timing.get("face_detect_s", 0.0)),
            "talknet_face_track_s": float(detect_stage_timing.get("face_track_s", 0.0)),
        }
        render_timing_table(ui["timing"], timing_stats)

        prob_fig = plot_probabilities(infer_result)
        ui["prob"].pyplot(prob_fig, use_container_width=True)
        plt.close(prob_fig)

        manifest = {
            "video_path": os.path.abspath(video_path),
            "detect_speaker_root": detect_root,
            "detect_speaker_path": detect_path,
            "wav_path": wav_path,
            "video_feats_pkl": video_feats_path,
            "audio_feats_pkl": audio_feats_path,
            "speaker_overlay_video": overlay_path,
            "web_video_path": overlay_path,
            "timing_stats": timing_stats,
        }
        manifest_path = os.path.join(PREPROCESSED_ROOT, "pipeline_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        ui["status"].success("Pipeline finished")
        render_result_banner(ui["result"], infer_result)
        return manifest, text, infer_result, timing_stats
    finally:
        torch_cleanup()


def reset_workspace():
    for path in _get_reset_cleanup_paths():
        if os.path.exists(path):
            shutil.rmtree(path)
        # Do not recreate RAM-disk detect root on reset.
        if path != os.path.join("/dev/shm", "mintrec_detect_speaker"):
            os.makedirs(path, exist_ok=True)


def torch_cleanup():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass
    gc.collect()


def release_runtime_caches(
    clear_inference_cache=False,
    clear_detector_cache=False,
    clear_aux_modal_cache=False,
):
    try:
        # Keep non-resident TalkNet/YuNet caches reclaimable on reset/exit.
        clear_feature_runtime_caches(clear_talknet=True, clear_yunet=True)
    except Exception:
        pass
    if clear_inference_cache:
        try:
            clear_inference_model_cache()
        except Exception:
            pass
        try:
            load_shared_inference_bundle.clear()
        except Exception:
            pass
    if clear_detector_cache:
        try:
            load_shared_detector_model.clear()
        except Exception:
            pass
    if clear_aux_modal_cache:
        try:
            load_shared_audio_feature_bundle.clear()
        except Exception:
            pass
        try:
            load_shared_whisper_tiny_model.clear()
        except Exception:
            pass
    try:
        plt.close("all")
    except Exception:
        pass
    torch_cleanup()
    # Return freed CPU heap pages to OS when possible (Linux/glibc).
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


def cleanup_on_exit():
    # Best-effort cleanup for abnormal exits (SIGINT/SIGTERM) and normal shutdown.
    release_runtime_caches(
        clear_inference_cache=True,
        clear_detector_cache=True,
        clear_aux_modal_cache=True,
    )
    for path in _get_reset_cleanup_paths():
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)


atexit.register(cleanup_on_exit)
# Streamlit may not allow signal handlers in its runtime; guard to avoid crash.
try:
    signal.signal(signal.SIGINT, lambda *_: cleanup_on_exit())
    signal.signal(signal.SIGTERM, lambda *_: cleanup_on_exit())
except Exception:
    pass


if "asr_text" not in st.session_state:
    st.session_state.asr_text = ""
if "infer_result" not in st.session_state:
    st.session_state.infer_result = None
if "speaker_overlay_video" not in st.session_state:
    st.session_state.speaker_overlay_video = None
if "web_video_path" not in st.session_state:
    st.session_state.web_video_path = None
if "wav_path" not in st.session_state:
    st.session_state.wav_path = None
if "timing_stats" not in st.session_state:
    st.session_state.timing_stats = None


# Preload and keep core models resident at app startup.
shared_detector_model = load_shared_detector_model()
shared_inference_bundle = load_shared_inference_bundle()
shared_audio_feature_bundle = load_shared_audio_feature_bundle()
shared_whisper_model = load_shared_whisper_tiny_model()

st.markdown(
    """
<style>
div.block-container {padding-top: 0.40rem; padding-bottom: 0.52rem; max-width: 97vw;}
h1, h2, h3 {margin-top: 0.12rem; margin-bottom: 0.12rem;}
.module-title {font-size: 1.42rem; font-weight: 760; margin: 0.1rem 0 0.24rem 0;}
.section-label {font-size: 0.98rem; font-weight: 660; margin: 0.08rem 0 0.14rem 0; color: #2e3b4e;}
.stVideo {max-height: 36vh; overflow: hidden; border-radius: 12px;}
.result-banner {
  background: linear-gradient(90deg, #0f4b8f 0%, #1b7a59 100%);
  color: #ffffff;
  padding: 0.62rem 0.9rem;
  border-radius: 12px;
  margin: 0.14rem 0 0.28rem 0;
  display: flex;
  align-items: center;
  gap: 0.65rem;
  flex-wrap: wrap;
}
.result-tag {
  background: rgba(255,255,255,0.2);
  border: 1px solid rgba(255,255,255,0.3);
  border-radius: 999px;
  padding: 0.15rem 0.58rem;
  font-size: 0.84rem;
  font-weight: 620;
}
.result-main {font-size: 1.03rem; font-weight: 740;}
.result-sep {opacity: 0.82;}
.text-box {
  background: #f6f8fb;
  border: 1px solid #dce3ef;
  border-radius: 10px;
  padding: 0.46rem 0.58rem;
  min-height: 56px;
  max-height: 104px;
  overflow-y: auto;
  line-height: 1.32;
}
.small-meta {margin-top: 0.12rem; color: #607084; font-size: 0.84rem;}
.timing-wrap {
  margin: 0.08rem 0 0.38rem 0;
  border: 1px solid #d5e3f2;
  border-radius: 12px;
  background: #f7fbff;
  overflow: hidden;
}
.timing-table {width: 100%; border-collapse: collapse; font-size: 0.90rem;}
.timing-table th {
  background: #e8f1fb;
  color: #1a3552;
  font-weight: 700;
  padding: 0.50rem 0.58rem;
  border-right: 1px solid #cfe0f2;
  text-align: center;
}
.timing-table td {
  padding: 0.52rem 0.58rem;
  border-top: 1px solid #dbe8f6;
  border-right: 1px solid #e1ecf8;
  text-align: center;
  font-weight: 600;
  color: #23476a;
}
.timing-table th:last-child, .timing-table td:last-child {border-right: none;}
</style>
""",
    unsafe_allow_html=True,
)

st.title("Z Single-Video Multimodal Dashboard")
head_left, head_right = st.columns([1.65, 1.0], gap="small")
with head_left:
    uploaded_video = st.file_uploader("Upload one video (.mp4)", type=["mp4"])

if uploaded_video is not None:
    save_uploaded_video(uploaded_video)

status_placeholder = st.empty()
st.markdown('<div class="module-title">耗时统计</div>', unsafe_allow_html=True)
timing_placeholder = st.empty()

upper_left, upper_right = st.columns([1.42, 1.0], gap="small")
with upper_left:
    st.markdown('<div class="module-title">Visual: Speaker BBox</div>', unsafe_allow_html=True)
    visual_video_placeholder = st.empty()
    visual_meta_placeholder = st.empty()

with upper_right:
    st.markdown('<div class="section-label">Audio: Waveform</div>', unsafe_allow_html=True)
    audio_plot_placeholder = st.empty()
    st.markdown('<div class="section-label">Text</div>', unsafe_allow_html=True)
    text_placeholder = st.empty()

result_placeholder = st.empty()
st.markdown('<div class="module-title">Intent Probabilities</div>', unsafe_allow_html=True)
prob_placeholder = st.empty()

web_video_path = st.session_state.web_video_path or st.session_state.speaker_overlay_video
if web_video_path and os.path.exists(web_video_path):
    visual_video_placeholder.video(web_video_path)
if st.session_state.wav_path and os.path.exists(st.session_state.wav_path):
    existing_wave_fig = plot_waveform(st.session_state.wav_path)
    audio_plot_placeholder.pyplot(existing_wave_fig, use_container_width=True)
    plt.close(existing_wave_fig)
if st.session_state.asr_text:
    text_placeholder.markdown(render_text_box(st.session_state.asr_text), unsafe_allow_html=True)
if st.session_state.infer_result is not None:
    render_result_banner(result_placeholder, st.session_state.infer_result)
    existing_prob_fig = plot_probabilities(st.session_state.infer_result)
    prob_placeholder.pyplot(existing_prob_fig, use_container_width=True)
    plt.close(existing_prob_fig)
if st.session_state.timing_stats is not None:
    render_timing_table(timing_placeholder, st.session_state.timing_stats)

col1, col2 = st.columns(2)

with col1:
    if st.button("Run Full Pipeline", use_container_width=True):
        if uploaded_video is None:
            st.error("Please upload one video first.")
            st.stop()
        st.session_state.asr_text = ""
        st.session_state.infer_result = None
        st.session_state.speaker_overlay_video = None
        st.session_state.web_video_path = None
        st.session_state.wav_path = None
        st.session_state.timing_stats = None

        result_placeholder.empty()

        ui_refs = {
            "status": status_placeholder,
            "timing": timing_placeholder,
            "result": result_placeholder,
            "video": visual_video_placeholder,
            "video_meta": visual_meta_placeholder,
            "audio": audio_plot_placeholder,
            "text": text_placeholder,
            "prob": prob_placeholder,
        }
        try:
            manifest, asr_text, infer_result, timing_stats = run_pipeline_with_live_visuals(
                INPUT_VIDEO_PATH,
                shared_detector_model,
                shared_inference_bundle,
                shared_audio_feature_bundle,
                shared_whisper_model,
                ui_refs,
            )
        except Exception as e:
            st.error("Feature extraction failed.\n{}".format(str(e)))
        else:
            st.session_state.asr_text = asr_text
            st.session_state.infer_result = infer_result
            st.session_state.speaker_overlay_video = manifest.get("speaker_overlay_video")
            st.session_state.web_video_path = manifest.get("web_video_path")
            st.session_state.wav_path = manifest.get("wav_path")
            st.session_state.timing_stats = timing_stats

with col2:
    if st.button("Reset", use_container_width=True):
        reset_workspace()
        st.session_state.asr_text = ""
        st.session_state.infer_result = None
        st.session_state.speaker_overlay_video = None
        st.session_state.web_video_path = None
        st.session_state.wav_path = None
        st.session_state.timing_stats = None
        result_placeholder.empty()
        timing_placeholder.empty()
        # Keep MULT/FasterRCNN resident across reset for faster next run.
        release_runtime_caches(clear_inference_cache=False, clear_detector_cache=False)
        st.success("Workspace reset completed.")
