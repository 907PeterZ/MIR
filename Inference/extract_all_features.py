import argparse
import importlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import gc
from mmdet.apis import init_detector, inference_detector
from scipy.io import wavfile
from transformers import Wav2Vec2Model, Wav2Vec2Processor


ROOT = os.path.dirname(os.path.abspath(__file__))
TALKNET_DIR = os.path.join(ROOT, "TalkNet_ASD")
TALKNET_PRETRAIN_MODEL = os.path.join(TALKNET_DIR, "pretrain_TalkSet.model")
MMDET_CONFIG_PATH = os.path.join(
    ROOT,
    "cache",
    "mmdetection",
    "configs",
    "faster_rcnn",
    "faster_rcnn_r50_fpn_1x_coco.py",
)
MMDET_CHECKPOINT_PATH = os.path.join(
    ROOT,
    "cache",
    "mmdetection",
    "checkpoints",
    "faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
)
FACE_YUNET_MODEL_PATH = os.path.join(ROOT, "cache", "face_detector", "face_detection_yunet_2023mar.onnx")
WAV2VEC_PATH = os.path.join(ROOT, "cache", "wav_model", "wav2vec2-base-960h")
SINGLE_CLIP_ID = "single_clip"
# Keep threshold above 0.5 for better precision/stability.
ACTIVE_SPEAKER_THRESHOLD = 0.55


def parse_args():
    parser = argparse.ArgumentParser(description="Single-video multimodal feature extraction.")
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Absolute or relative path to one input .mp4 file.",
    )
    parser.add_argument(
        "--preprocessed_root",
        type=str,
        default=os.path.join(ROOT, "preprocessed_output"),
        help="Output root for intermediate and final files.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove preprocessed_root before running.",
    )
    return parser.parse_args()


def _save_pickle(path, value):
    with open(path, "wb") as f:
        pickle.dump(value, f)


def load_fasterrcnn_model(config_path=MMDET_CONFIG_PATH, checkpoint_path=MMDET_CHECKPOINT_PATH, device=None):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but GPU-only mode is enabled.")
    if device is None:
        device = "cuda:0"
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
    model = init_detector(config_path, checkpoint_path, device=device)
    # Warm up detector with file-path input to keep file inference pipeline stable.
    dummy = np.zeros((240, 320, 3), dtype=np.uint8)
    warmup_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            warmup_path = tmp.name
        cv2.imwrite(warmup_path, dummy)
        inference_detector(model, warmup_path)
    finally:
        if warmup_path and os.path.exists(warmup_path):
            os.remove(warmup_path)
    return model


def _get_ram_root():
    shm_root = "/dev/shm"
    if os.path.isdir(shm_root) and os.access(shm_root, os.W_OK):
        return shm_root
    return None


def _import_runtime_module(module_name):
    # Keep imported modules resident across runs to avoid repeated reload overhead.
    if module_name in sys.modules:
        return sys.modules[module_name]
    return importlib.import_module(module_name)


def run_detect_speaker(
    video_path,
    detect_root,
    detector_model=None,
    face_detector_weights=FACE_YUNET_MODEL_PATH,
    face_det_conf=0.8,
    face_det_nms=0.3,
    face_det_topk=5000,
    face_redetect_interval=2,
    face_track_person_iou=0.5,
    face_track_min_conf=0.85,
):
    save_path = os.path.join(detect_root, SINGLE_CLIP_ID)
    os.makedirs(detect_root, exist_ok=True)

    cwd = os.getcwd()
    try:
        os.chdir(TALKNET_DIR)
        if TALKNET_DIR not in sys.path:
            sys.path.insert(0, TALKNET_DIR)

        detect_speaker_mod = _import_runtime_module("detect_speaker")
        run_single_video = detect_speaker_mod.run_single_video

        print("Step 1/3: TalkNet active speaker detection...")
        run_single_video(
            video_file_path=video_path,
            save_path=save_path,
            pretrain_model=TALKNET_PRETRAIN_MODEL,
            config_file=MMDET_CONFIG_PATH,
            checkpoint_file=MMDET_CHECKPOINT_PATH,
            detector_model=detector_model,
            speaker_threshold=ACTIVE_SPEAKER_THRESHOLD,
            face_detector_weights=face_detector_weights,
            face_det_conf=face_det_conf,
            face_det_nms=face_det_nms,
            face_det_topk=face_det_topk,
            face_redetect_interval=face_redetect_interval,
            face_track_person_iou=face_track_person_iou,
            face_track_min_conf=face_track_min_conf,
        )
    finally:
        os.chdir(cwd)

    return detect_root, save_path


def run_video_feature_extraction(preprocessed_root, detect_root, detector_model=None):
    print("Step 2/3: Video ROI feature extraction...")
    video_preprocess_mod = _import_runtime_module("tools.video_preprocess")
    VideoFeature = video_preprocess_mod.VideoFeature

    args = SimpleNamespace(
        detection_checkpoint_path=MMDET_CHECKPOINT_PATH,
        detection_config_path=MMDET_CONFIG_PATH,
        video_data_path=preprocessed_root,
        video_feats_path="video_feats_all.pkl",
        frames_path="",
        speaker_annotation_path="",
        TalkNet_speaker_path=detect_root,
        use_TalkNet=True,
        roi_feat_size=7,
        frame_stride=1,
        roi_batch_size=2,
        max_frames=0,
        log_every=10,
    )

    video_data = VideoFeature(args, detector_model=detector_model)
    video_data._get_feats(args)

    if SINGLE_CLIP_ID not in video_data.bbox_feats:
        if len(video_data.bbox_feats) != 1:
            raise KeyError("Failed to locate single clip video features.")
        video_feats_single = next(iter(video_data.bbox_feats.values()))
    else:
        video_feats_single = video_data.bbox_feats[SINGLE_CLIP_ID]

    video_feats_path = os.path.join(preprocessed_root, "video_feats.pkl")
    _save_pickle(video_feats_path, video_feats_single)
    return video_feats_path


def load_wav2vec_feature_bundle(device="cpu"):
    requested_device = str(device).lower() if device is not None else "auto"
    use_cuda = requested_device in ("cuda", "cuda:0", "auto") and torch.cuda.is_available()
    runtime_device = torch.device("cuda:0" if use_cuda else "cpu")
    processor = Wav2Vec2Processor.from_pretrained(WAV2VEC_PATH)
    model = Wav2Vec2Model.from_pretrained(WAV2VEC_PATH)
    model = model.to(runtime_device)
    model.eval()
    return processor, model, runtime_device


def _extract_audio_features_from_wav(wav_path, preprocessed_root, device="cuda", audio_bundle=None):
    audio_feats_path = os.path.join(preprocessed_root, "audio_feats.pkl")
    owns_bundle = audio_bundle is None
    if owns_bundle:
        processor, model, runtime_device = load_wav2vec_feature_bundle(device=device)
    else:
        if len(audio_bundle) == 3:
            processor, model, runtime_device = audio_bundle
        else:
            processor, model = audio_bundle
            runtime_device = next(model.parameters()).device

    sr, y = wavfile.read(wav_path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    if np.max(np.abs(y)) > 0:
        y = y / max(np.max(np.abs(y)), 1.0)

    input_values = processor(y, sampling_rate=sr, return_tensors="pt").input_values.to(runtime_device)
    with torch.no_grad():
        audio_feats = model(input_values).last_hidden_state.squeeze(0).detach().cpu()

    _save_pickle(audio_feats_path, audio_feats)
    if owns_bundle:
        # Explicitly release audio model/processor after extraction when not shared.
        del model
        del processor
        if runtime_device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    return audio_feats_path


def run_audio_feature_extraction_from_wav(
    source_wav_path,
    preprocessed_root,
    device="cuda",
    audio_bundle=None,
    copy_to_preprocessed=True,
):
    print("Audio extraction and wav2vec2 features...")
    source_wav_path = os.path.abspath(source_wav_path)
    if not os.path.exists(source_wav_path):
        raise FileNotFoundError("Missing wav file: {}".format(source_wav_path))
    os.makedirs(preprocessed_root, exist_ok=True)
    target_wav_path = os.path.join(preprocessed_root, "raw_audio.wav")
    if copy_to_preprocessed or os.path.abspath(target_wav_path) != source_wav_path:
        shutil.copyfile(source_wav_path, target_wav_path)
    else:
        target_wav_path = source_wav_path

    audio_feats_path = _extract_audio_features_from_wav(
        wav_path=target_wav_path,
        preprocessed_root=preprocessed_root,
        device=device,
        audio_bundle=audio_bundle,
    )
    return audio_feats_path, target_wav_path


def run_audio_feature_extraction(video_path, preprocessed_root, device="cuda", audio_bundle=None):
    print("Audio extraction and wav2vec2 features...")
    wav_path = os.path.join(preprocessed_root, "raw_audio.wav")

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-ac",
        "1",
        "-vn",
        "-ar",
        "16000",
        wav_path,
        "-loglevel",
        "error",
    ]
    subprocess.check_call(ffmpeg_cmd)

    audio_feats_path = _extract_audio_features_from_wav(
        wav_path=wav_path,
        preprocessed_root=preprocessed_root,
        device=device,
        audio_bundle=audio_bundle,
    )
    return audio_feats_path, wav_path


def clear_feature_runtime_caches(clear_talknet=True, clear_yunet=True):
    cwd = os.getcwd()
    try:
        os.chdir(TALKNET_DIR)
        if TALKNET_DIR not in sys.path:
            sys.path.insert(0, TALKNET_DIR)
        detect_speaker_mod = _import_runtime_module("detect_speaker")
        clear_fn = getattr(detect_speaker_mod, "clear_runtime_model_caches", None)
        if callable(clear_fn):
            clear_fn(clear_talknet=clear_talknet, clear_yunet=clear_yunet)
    finally:
        os.chdir(cwd)


def run_single_video_pipeline(
    video_path,
    preprocessed_root,
    clean=False,
    detector_model=None,
    person_detector_model=None,
    use_ram=True,
):
    video_path = os.path.abspath(video_path)
    preprocessed_root = os.path.abspath(preprocessed_root)

    if not os.path.exists(video_path):
        raise FileNotFoundError("Input video not found: {}".format(video_path))

    if clean and os.path.exists(preprocessed_root):
        shutil.rmtree(preprocessed_root)
    os.makedirs(preprocessed_root, exist_ok=True)

    # FasterRCNN is still used for ROI feature extraction to keep training/inference consistency.
    roi_detector = detector_model
    if roi_detector is None:
        roi_detector = load_fasterrcnn_model()

    # Reuse FasterRCNN for front person detection to avoid two different detector stacks.
    front_person_detector = person_detector_model if person_detector_model is not None else roi_detector

    ram_root = _get_ram_root() if use_ram else None
    if ram_root is not None:
        detect_root = os.path.join(ram_root, "mintrec_detect_speaker")
    else:
        detect_root = os.path.join(preprocessed_root, "detect_speaker")

    detect_root, detect_path = run_detect_speaker(video_path, detect_root, detector_model=front_person_detector)
    video_feats_path = run_video_feature_extraction(preprocessed_root, detect_root, detector_model=roi_detector)
    talknet_wav_path = os.path.join(detect_path, "pyavi", "audio.wav")
    if os.path.exists(talknet_wav_path):
        audio_feats_path, wav_path = run_audio_feature_extraction_from_wav(
            talknet_wav_path,
            preprocessed_root,
            copy_to_preprocessed=True,
        )
    else:
        audio_feats_path, wav_path = run_audio_feature_extraction(video_path, preprocessed_root)
    # Free RAM disk after all consumers have finished reading detect outputs.
    if ram_root is not None and os.path.isdir(detect_root):
        shutil.rmtree(detect_root, ignore_errors=True)

    manifest = {
        "video_path": video_path,
        "detect_speaker_root": detect_root,
        "detect_speaker_path": detect_path,
        "wav_path": wav_path,
        "video_feats_pkl": video_feats_path,
        "audio_feats_pkl": audio_feats_path,
    }
    manifest_path = os.path.join(preprocessed_root, "pipeline_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Cleanup local tensors and free GPU cache if any.
    torch.cuda.empty_cache()
    gc.collect()
    return manifest


if __name__ == "__main__":
    args = parse_args()
    manifest = run_single_video_pipeline(
        video_path=args.video_path,
        preprocessed_root=args.preprocessed_root,
        clean=args.clean,
        detector_model=None,
    )
    manifest_path = os.path.join(os.path.abspath(args.preprocessed_root), "pipeline_manifest.json")

    print("Feature extraction completed.")
    print("Manifest saved to: {}".format(manifest_path))
