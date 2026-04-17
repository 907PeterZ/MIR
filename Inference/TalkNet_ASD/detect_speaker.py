from genericpath import exists
import sys
import time
import os
import tqdm
import torch
import argparse
import glob
import subprocess
import warnings
import cv2
import pickle
import pdb
import math
import python_speech_features
import json
import traceback
import copy
import urllib.request
import gc

from scipy import signal
from shutil import rmtree
from scipy.io import wavfile
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, f1_score

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

try:
    from talkNet import talkNet
except ImportError:
    from TalkNet_ASD.talkNet import talkNet

import numpy as np

from mmdet.apis import init_detector, inference_detector
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.core import encode_mask_results
from mmdet.core.visualization import imshow_det_bboxes
from mmcv.parallel import collate, scatter
from mmcv.ops import RoIPool


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
DEFAULT_MMDET_CONFIG = os.path.join(
    PROJECT_ROOT,
    "cache",
    "mmdetection",
    "configs",
    "faster_rcnn",
    "faster_rcnn_r50_fpn_1x_coco.py",
)
DEFAULT_MMDET_CHECKPOINT = os.path.join(
    PROJECT_ROOT,
    "cache",
    "mmdetection",
    "checkpoints",
    "faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
)
DEFAULT_FACE_DETECTOR_WEIGHTS = os.path.join(
    PROJECT_ROOT,
    "cache",
    "face_detector",
    "face_detection_yunet_2023mar.onnx",
)
DEFAULT_FACE_DETECTOR_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
DEFAULT_ACTIVE_SPEAKER_THRESHOLD = 0.55
_YUNET_DETECTOR_CACHE = {}
_TALKNET_RUNTIME_CACHE = {}


def process_video(read_file_path, output_file_path, max_width=None):

    cap = cv2.VideoCapture(read_file_path)

    if not cap.isOpened():
        print('The directory is wrong.')
    
    cnt = 0
    while True:

        ret, frame = cap.read()
        if frame is None:
            break
        
        if max_width and frame.shape[1] > max_width:
            scale = max_width / float(frame.shape[1])
            new_h = int(frame.shape[0] * scale)
            frame = cv2.resize(frame, (max_width, new_h), interpolation=cv2.INTER_AREA)

        write_path = os.path.join(output_file_path, str('%06d' % cnt) + '.jpg')
        cv2.imwrite(write_path, frame)
        cnt += 1

        if not ret:
            break


def _build_test_pipeline(model, load_from_webcam):
    cfg = copy.deepcopy(model.cfg)
    if load_from_webcam:
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
    else:
        cfg.data.test.pipeline[0].type = 'LoadImageFromFile'
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    return Compose(cfg.data.test.pipeline)


def _test_with_pipeline(model, imgs, test_pipeline):
    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    device = next(model.parameters()).device  # model device

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        try:
            data = test_pipeline(data)
        except Exception as e:
            raise RuntimeError(
                "mmdet test pipeline failed: first_step={}, input_type={}, keys={}".format(
                    "LoadImageFromWebcam" if isinstance(img, np.ndarray) else "LoadImageFromFile",
                    type(img).__name__,
                    list(data.keys()),
                )
            ) from e
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]

    if next(model.parameters()).is_cuda:
        # scatter to specified GPU (mmcv expects device ids as ints)
        device_id = device.index if device.index is not None else 0
        data = scatter(data, [device_id])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)

    if not is_batch:
        return results[0]
    else:
        return results


def test(model, imgs):
    if isinstance(imgs, (list, tuple)):
        first = imgs[0]
    else:
        first = imgs
    test_pipeline = _build_test_pipeline(model, isinstance(first, np.ndarray))
    return _test_with_pipeline(model, imgs, test_pipeline)


def _iter_batches(sequence, batch_size):
    batch_size = max(1, int(batch_size))
    for start in range(0, len(sequence), batch_size):
        yield start, sequence[start:start + batch_size]


def _ensure_face_detector_weights(weights_path):
    if os.path.exists(weights_path):
        return weights_path
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    sys.stderr.write("Downloading YuNet face detector weights to {}...\n".format(weights_path))
    tmp_path = weights_path + ".tmp"
    urllib.request.urlretrieve(DEFAULT_FACE_DETECTOR_URL, tmp_path)
    os.replace(tmp_path, weights_path)
    return weights_path


def _load_yunet_face_detector(weights_path, score_th=0.8, nms_th=0.3, top_k=5000):
    if not hasattr(cv2, "FaceDetectorYN"):
        raise RuntimeError("Current OpenCV build does not provide FaceDetectorYN (YuNet).")
    model_path = _ensure_face_detector_weights(weights_path)
    cache_key = (
        os.path.abspath(model_path),
        float(score_th),
        float(nms_th),
        int(top_k),
    )
    detector = _YUNET_DETECTOR_CACHE.get(cache_key)
    if detector is None:
        detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            (320, 320),
            float(score_th),
            float(nms_th),
            int(top_k),
        )
        _YUNET_DETECTOR_CACHE[cache_key] = detector
    return detector


def _get_cached_talknet_runtime(pretrain_model, device):
    cache_key = (os.path.abspath(pretrain_model), str(device))
    runtime = _TALKNET_RUNTIME_CACHE.get(cache_key)
    if runtime is None:
        runtime = talkNet(device=device)
        runtime.loadParameters(pretrain_model)
        runtime.eval()
        _TALKNET_RUNTIME_CACHE[cache_key] = runtime
        sys.stderr.write("Model %s loaded from previous state! \r\n" % pretrain_model)
    return runtime


def clear_runtime_model_caches(clear_talknet=True, clear_yunet=True):
    if clear_talknet:
        _TALKNET_RUNTIME_CACHE.clear()
    if clear_yunet:
        _YUNET_DETECTOR_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    gc.collect()


def _detect_faces_yunet(detector, person_image_bgr, conf_th=0.8, max_side=320, cache_state=None):
    h, w = person_image_bgr.shape[:2]
    if h < 20 or w < 20:
        return []

    scale = 1.0
    detect_img = person_image_bgr
    if max_side and max_side > 0:
        max_hw = max(h, w)
        if max_hw > max_side:
            scale = float(max_side) / float(max_hw)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            detect_img = cv2.resize(person_image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    in_h, in_w = detect_img.shape[:2]
    input_size = (in_w, in_h)
    if cache_state is None:
        detector.setInputSize(input_size)
    else:
        prev_size = cache_state.get("input_size")
        if prev_size != input_size:
            detector.setInputSize(input_size)
            cache_state["input_size"] = input_size

    _, faces = detector.detect(detect_img)
    if faces is None or len(faces) == 0:
        return []

    outputs = []
    inv_scale = 1.0 / scale
    for face in faces:
        score = float(face[14]) if len(face) > 14 else float(face[4])
        if score < conf_th:
            continue
        x, y, bw, bh = [float(v) for v in face[:4]]
        x1 = max(0.0, x * inv_scale)
        y1 = max(0.0, y * inv_scale)
        x2 = min(float(w), (x + bw) * inv_scale)
        y2 = min(float(h), (y + bh) * inv_scale)
        if x2 <= x1 or y2 <= y1:
            continue
        outputs.append(np.array([x1, y1, x2, y2, score], dtype=np.float32))
    return outputs


def _bbox_iou_xyxy(box_a, box_b):
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(0.0, float(box_b[3]) - float(box_b[1]))
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return float(inter_area / union)


def _clip_bbox_xyxy(box, image_w, image_h):
    x1 = max(0.0, min(float(image_w), float(box[0])))
    y1 = max(0.0, min(float(image_h), float(box[1])))
    x2 = max(0.0, min(float(image_w), float(box[2])))
    y2 = max(0.0, min(float(image_h), float(box[3])))
    if x2 <= x1 or y2 <= y1:
        return None
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _face_bbox_to_person_rel(face_bbox, person_bbox):
    pw = float(person_bbox[2]) - float(person_bbox[0])
    ph = float(person_bbox[3]) - float(person_bbox[1])
    if pw <= 1e-6 or ph <= 1e-6:
        return None
    rel = np.array(
        [
            (float(face_bbox[0]) - float(person_bbox[0])) / pw,
            (float(face_bbox[1]) - float(person_bbox[1])) / ph,
            (float(face_bbox[2]) - float(person_bbox[0])) / pw,
            (float(face_bbox[3]) - float(person_bbox[1])) / ph,
        ],
        dtype=np.float32,
    )
    rel[0] = np.clip(rel[0], 0.0, 1.0)
    rel[1] = np.clip(rel[1], 0.0, 1.0)
    rel[2] = np.clip(rel[2], 0.0, 1.0)
    rel[3] = np.clip(rel[3], 0.0, 1.0)
    if rel[2] <= rel[0] or rel[3] <= rel[1]:
        return None
    return rel


def _project_face_bbox_from_rel(rel_bbox, person_bbox, image_w, image_h):
    pw = float(person_bbox[2]) - float(person_bbox[0])
    ph = float(person_bbox[3]) - float(person_bbox[1])
    if pw <= 1e-6 or ph <= 1e-6:
        return None
    face_bbox = np.array(
        [
            float(person_bbox[0]) + float(rel_bbox[0]) * pw,
            float(person_bbox[1]) + float(rel_bbox[1]) * ph,
            float(person_bbox[0]) + float(rel_bbox[2]) * pw,
            float(person_bbox[1]) + float(rel_bbox[3]) * ph,
        ],
        dtype=np.float32,
    )
    return _clip_bbox_xyxy(face_bbox, image_w=image_w, image_h=image_h)


def _find_best_prev_face_entry(person_bbox, prev_face_entries, min_iou):
    best_entry = None
    best_iou = 0.0
    for entry in prev_face_entries:
        iou = _bbox_iou_xyxy(person_bbox, entry["person_bbox"])
        if iou > best_iou:
            best_iou = iou
            best_entry = entry
    if best_entry is None or best_iou < float(min_iou):
        return None, best_iou
    return best_entry, best_iou


def get_bbox_data(args, detector_model=None):
    model = detector_model
    if model is None:
        model = init_detector(args.config_file, args.checkpoint_file, device=args.device)

    classes = getattr(model, "CLASSES", None)
    map_class = {i: v for i, v in enumerate(classes)} if classes is not None else None

    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
    flist.sort()
    dets = []
    det_batch_size = max(1, int(getattr(args, "det_batch_size", 2)))
    log_every_frames = max(1, int(getattr(args, "log_every_frames", 20)))
    test_pipeline = _build_test_pipeline(model, load_from_webcam=False)
    for start, batch_files in _iter_batches(flist, det_batch_size):
        batch_results = _test_with_pipeline(model, batch_files, test_pipeline)
        if not isinstance(batch_results, (list, tuple)):
            batch_results = [batch_results]

        for local_idx, result in enumerate(batch_results):
            fidx = start + local_idx
            dets.append([])
            for idx, elem in enumerate(result):
                if map_class is None:
                    is_person = (idx == 0)
                else:
                    label = map_class[idx]
                    is_person = (label == 'person')

                if is_person:
                    for bbox in elem:
                        score = bbox[-1]
                        if score < 0.8:
                            continue
                        dets[-1].append({'frame':fidx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]}) # dets has the frames info, bbox info, conf info
            if (fidx % log_every_frames == 0) or (fidx == len(flist) - 1):
                sys.stderr.write('%s-%05d; %d dets\r' % (args.videoFilePath, fidx, len(dets[-1])))
        #bbox: [lh, lw, rh, rw]        
    return dets

def scene_detect(args):
    # CPU: Scene detection, output is the list of each shot's time duration
    videoManager = VideoManager([args.videoFilePath])
    statsManager = StatsManager()
    sceneManager = SceneManager(statsManager)
    sceneManager.add_detector(ContentDetector())
    baseTimecode = videoManager.get_base_timecode()
    videoManager.set_downscale_factor()
    videoManager.start()
    sceneManager.detect_scenes(frame_source = videoManager)
    sceneList = sceneManager.get_scene_list(baseTimecode)
    savePath = os.path.join(args.pyworkPath, 'scene.pckl')
    if sceneList == []:
        sceneList = [(videoManager.get_base_timecode(),videoManager.get_current_timecode())]
    
    return sceneList

def inference_video(args, persons):
    # GPU: Face detection, output is the list contains the face location and score in this frame
    face_conf_th = float(getattr(args, "face_det_conf", 0.8))
    face_det_max_side = int(getattr(args, "face_det_max_side", 320))
    face_redetect_interval = max(1, int(getattr(args, "face_redetect_interval", 2)))
    face_track_person_iou = float(getattr(args, "face_track_person_iou", 0.5))
    face_track_min_conf = float(getattr(args, "face_track_min_conf", 0.85))
    min_person_side = int(getattr(args, "min_person_side", 150))
    DET = _load_yunet_face_detector(
        getattr(args, "face_detector_weights", DEFAULT_FACE_DETECTOR_WEIGHTS),
        score_th=face_conf_th,
        nms_th=float(getattr(args, "face_det_nms", 0.3)),
        top_k=int(getattr(args, "face_det_topk", 5000)),
    )
    yunet_cache = {"input_size": None}

    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
    flist.sort()
    dets = []
    log_every_frames = max(1, int(getattr(args, "log_every_frames", 20)))
    max_persons_per_frame = int(getattr(args, "max_persons_per_frame", 5))
    face_detector_calls = 0
    face_track_reuse = 0
    prev_face_entries = []
    for fidx, fname in enumerate(flist):
        dets.append([])
        frame_persons = persons[fidx]
        force_redetect_frame = (fidx % face_redetect_interval == 0)
        next_prev_face_entries = []
        if max_persons_per_frame > 0 and len(frame_persons) > max_persons_per_frame:
            frame_persons = sorted(frame_persons, key=lambda p: float(p.get('conf', 0.0)), reverse=True)[:max_persons_per_frame]
        if len(frame_persons) == 0:
            prev_face_entries = []
            if (fidx % log_every_frames == 0) or (fidx == len(flist) - 1):
                sys.stderr.write('%s-%05d; %d dets\r' % (args.videoFilePath, fidx, len(dets[-1])))
            continue

        image = cv2.imread(fname)
        if image is None:
            prev_face_entries = []
            if (fidx % log_every_frames == 0) or (fidx == len(flist) - 1):
                sys.stderr.write('%s-%05d; %d dets\r' % (args.videoFilePath, fidx, len(dets[-1])))
            continue
        image_h, image_w = image.shape[:2]

        for person in frame_persons:
            personBbox = person['bbox']
            px1 = max(0, int(personBbox[0]))
            py1 = max(0, int(personBbox[1]))
            px2 = min(image_w, int(personBbox[2]))
            py2 = min(image_h, int(personBbox[3]))
            if (px2 - px1) < min_person_side or (py2 - py1) < min_person_side:
                continue
            person_bbox = np.array([px1, py1, px2, py2], dtype=np.float32)

            tracked_face_bbox = None
            tracked_face_conf = None
            if len(prev_face_entries) > 0:
                prev_entry, _ = _find_best_prev_face_entry(
                    person_bbox,
                    prev_face_entries,
                    min_iou=face_track_person_iou,
                )
                if prev_entry is not None and float(prev_entry.get("conf", 0.0)) >= face_track_min_conf:
                    projected = _project_face_bbox_from_rel(
                        prev_entry["face_rel"],
                        person_bbox,
                        image_w=image_w,
                        image_h=image_h,
                    )
                    if projected is not None:
                        pw = float(person_bbox[2] - person_bbox[0])
                        ph = float(person_bbox[3] - person_bbox[1])
                        fw = float(projected[2] - projected[0])
                        fh = float(projected[3] - projected[1])
                        if fw >= max(12.0, 0.05 * pw) and fh >= max(12.0, 0.05 * ph):
                            tracked_face_bbox = projected
                            tracked_face_conf = float(prev_entry["conf"])

            selected_face_bbox = None
            selected_face_conf = None
            if force_redetect_frame or tracked_face_bbox is None:
                person_image = image[py1:py2, px1:px2]
                bboxes = _detect_faces_yunet(
                    DET,
                    person_image,
                    conf_th=face_conf_th,
                    max_side=face_det_max_side,
                    cache_state=yunet_cache,
                )
                face_detector_calls += 1
                idx = -1
                s = -1
                for i, bbox in enumerate(bboxes):
                    if bbox[4] > s:
                        idx = i
                        s = bbox[4]
                if idx != -1:
                    bbox = bboxes[idx]
                    selected_face_bbox = np.array(
                        [
                            float(bbox[0]) + px1,
                            float(bbox[1]) + py1,
                            float(bbox[2]) + px1,
                            float(bbox[3]) + py1,
                        ],
                        dtype=np.float32,
                    )
                    selected_face_conf = float(bbox[4])

            if selected_face_bbox is None and tracked_face_bbox is not None:
                selected_face_bbox = tracked_face_bbox
                selected_face_conf = tracked_face_conf
                face_track_reuse += 1

            if selected_face_bbox is None:
                continue

            dets[-1].append(
                {
                    'frame': fidx,
                    'bbox': selected_face_bbox.tolist(),
                    'person_bbox': personBbox,
                    'conf': selected_face_conf,
                }
            )  # dets has the frames info, bbox info, conf info

            face_rel = _face_bbox_to_person_rel(selected_face_bbox, person_bbox)
            if face_rel is not None:
                next_prev_face_entries.append(
                    {
                        "person_bbox": person_bbox,
                        "face_rel": face_rel,
                        "conf": selected_face_conf,
                    }
                )
        if (fidx % log_every_frames == 0) or (fidx == len(flist) - 1):
            sys.stderr.write('%s-%05d; %d dets\r' % (args.videoFilePath, fidx, len(dets[-1])))
        prev_face_entries = next_prev_face_entries

    sys.stderr.write(
        "\nFace detector calls: {} | Tracked reuse: {} | Redetect interval: {}\n".format(
            face_detector_calls,
            face_track_reuse,
            face_redetect_interval,
        )
    )
    
    return dets

def bb_intersection_over_union(boxA, boxB, evalCol = False):
    # CPU: IOU Function to calculate overlap between two image
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if evalCol == True:
        iou = interArea / float(boxAArea)
    else:
        iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def track_shot(args, sceneFaces):
    # CPU: Face tracking
    iouThres  = 0.5     # Minimum IOU between consecutive face detections
    tracks    = []
    while True:
        track     = []
        for frameFaces in sceneFaces:
            for face in frameFaces:
                if track == []:
                    track.append(face)
                    frameFaces.remove(face)
                elif face['frame'] - track[-1]['frame'] <= args.numFailedDet:
                    iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
                    if iou > iouThres:
                        track.append(face)
                        frameFaces.remove(face)
                        continue
                else:
                    break
        if track == []:
            break
        elif len(track) > args.minTrack:
            frameNum    = np.array([ f['frame'] for f in track ])
            bboxes      = np.array([np.array(f['bbox']) for f in track])
            personBboxes = np.array([np.array(f['person_bbox']) for f in track])
            frameI      = np.arange(frameNum[0],frameNum[-1]+1)
            bboxesI    = []
            personBboxesI = []
            for ij in range(0,4):
                interpfn  = interp1d(frameNum, bboxes[:,ij])
                bboxesI.append(interpfn(frameI))
                personInterpfn = interp1d(frameNum, personBboxes[:,ij])
                personBboxesI.append(personInterpfn(frameI))
            bboxesI  = np.stack(bboxesI, axis=1)
            personBboxesI = np.stack(personBboxesI, axis=1)
            print('track')
            print(track)
            if max(np.mean(bboxesI[:,2]-bboxesI[:,0]), np.mean(bboxesI[:,3]-bboxesI[:,1])) > args.minFaceSize:
                tracks.append({'frame':frameI,'bbox':bboxesI, 'person_bbox': personBboxesI})
    
    return tracks

def crop_video(args, track, cropFile):
    # CPU: crop the face clips
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg')) # Read the frames
    flist.sort()
    vOut = cv2.VideoWriter(cropFile + 't.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (224,224))# Write video
    dets = {'x':[], 'y':[], 's':[]}
    personDets = {'x':[], 'y':[], 's':[]}
    for det in track['bbox']: # Read the tracks
        dets['s'].append(max((det[3]-det[1]), (det[2]-det[0]))/2) 
        dets['y'].append((det[1]+det[3])/2) # crop center x 
        dets['x'].append((det[0]+det[2])/2) # crop center y
    dets['s'] = signal.medfilt(dets['s'], kernel_size=13)  # Smooth detections 
    dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
    dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
    for det in track['person_bbox']: # Read the tracks
        personDets['s'].append(max((det[3]-det[1]), (det[2]-det[0]))/2) 
        personDets['y'].append((det[1]+det[3])/2) # crop center x 
        personDets['x'].append((det[0]+det[2])/2) # crop center y
    personDets['s'] = signal.medfilt(personDets['s'], kernel_size=13)  # Smooth detections 
    personDets['x'] = signal.medfilt(personDets['x'], kernel_size=13)
    personDets['y'] = signal.medfilt(personDets['y'], kernel_size=13)
    for fidx, frame in enumerate(track['frame']):
        cs  = args.cropScale
        bs  = dets['s'][fidx]   # Detection box size
        bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount 
        image = cv2.imread(flist[frame])
        frame = np.pad(image, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
        my  = dets['y'][fidx] + bsi  # BBox center Y
        mx  = dets['x'][fidx] + bsi  # BBox center X
        face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
        
        
        vOut.write(cv2.resize(face, (224, 224)))
    audioTmp    = cropFile + '.wav'
    audioStart  = (track['frame'][0]) / 25
    audioEnd    = (track['frame'][-1]+1) / 25
    vOut.release()
    command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" % \
              (args.audioFilePath, args.nDataLoaderThread, audioStart, audioEnd, audioTmp)) 
    output = subprocess.call(command, shell=True, stdout=None) # Crop audio file
    _, audio = wavfile.read(audioTmp)
    command = ("ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic" % \
              (cropFile, audioTmp, args.nDataLoaderThread, cropFile)) # Combine audio and video file
    output = subprocess.call(command, shell=True, stdout=None)
    os.remove(cropFile + 't.avi')
    return {'track':track, 'proc_track':dets, 'person_proc_track': personDets}

def extract_MFCC(file, outPath):
    # CPU: extract mfcc
    sr, audio = wavfile.read(file)
    mfcc = python_speech_features.mfcc(audio,sr) # (N_frames, 13)   [1s = 100 frames]
    featuresPath = os.path.join(outPath, file.split('/')[-1].replace('.wav', '.npy'))
    np.save(featuresPath, mfcc)

def evaluate_network(files, args):
    # GPU: active speaker detection by pretrained TalkNet
    s = _get_cached_talknet_runtime(args.pretrainModel, args.device)
    allScores = []
    device = s.device
    # durationSet = {1,2,4,6} # To make the result more reliable
    durationSet = {1,1,1,2,2,2,3,3,4,5,6} # Use this line can get more reliable result
    for file in tqdm.tqdm(files, total = len(files)):
        fileName = os.path.splitext(file.split('/')[-1])[0] # Load audio and video
        _, audio = wavfile.read(os.path.join(args.pycropPath, fileName + '.wav'))
        audioFeature = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025, winstep = 0.010)
        video = cv2.VideoCapture(os.path.join(args.pycropPath, fileName + '.avi'))
        videoFeature = []
        while video.isOpened():
            ret, frames = video.read()
            if ret == True:
                face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (224,224))
                face = face[int(112-(112/2)):int(112+(112/2)), int(112-(112/2)):int(112+(112/2))]
                videoFeature.append(face)
            else:
                break
        video.release()
        videoFeature = np.array(videoFeature)
        length = min((audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100, videoFeature.shape[0])
        audioFeature = audioFeature[:int(round(length * 100)),:]
        videoFeature = videoFeature[:int(round(length * 25)),:,:]
        allScore = [] # Evaluation use TalkNet
        for duration in durationSet:
            batchSize = int(math.ceil(length / duration))
            scores = []
            with torch.no_grad():
                for i in range(batchSize):
                    inputA = torch.FloatTensor(audioFeature[i * duration * 100:(i+1) * duration * 100,:]).unsqueeze(0).to(device)
                    inputV = torch.FloatTensor(videoFeature[i * duration * 25: (i+1) * duration * 25,:,:]).unsqueeze(0).to(device)
                    embedA = s.model.forward_audio_frontend(inputA)
                    embedV = s.model.forward_visual_frontend(inputV)	
                    embedA, embedV = s.model.forward_cross_attention(embedA, embedV)
                    out = s.model.forward_audio_visual_backend(embedA, embedV)
                    # Convert TalkNet logits to speaking probability [0, 1].
                    logits = s.lossAV.FC(out.squeeze(1))
                    if logits.dim() == 3:
                        logits = logits.reshape(-1, logits.size(-1))
                    prob = torch.softmax(logits, dim=-1)[:, 1]
                    scores.extend(prob.detach().cpu().numpy().tolist())
            allScore.append(scores)
        allScore = np.round((np.mean(np.array(allScore), axis = 0)), 1).astype(float)
        allScores.append(allScore)	
    return allScores

def visualization(tracks, scores, args, render_video=True):
    # CPU: compute best_persons; optionally render video.
    asd_threshold = float(getattr(args, "activeSpeakerThreshold", DEFAULT_ACTIVE_SPEAKER_THRESHOLD))
    flist = glob.glob(os.path.join(args.pyframesPath, '*.jpg'))
    flist.sort()
    faces = [[] for i in range(len(flist))]
    for tidx, track in enumerate(tracks):
        score = scores[tidx]
        for fidx, frame in enumerate(track['track']['frame'].tolist()):
            s = score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)] # average smoothing
            s = np.mean(s)
            faces[frame].append({'track':tidx, 'score':float(s),'s':track['person_proc_track']['s'][fidx], 'x':track['person_proc_track']['x'][fidx], 'y':track['person_proc_track']['y'][fidx],
                        's_':track['proc_track']['s'][fidx], 'x_':track['proc_track']['x'][fidx], 'y_':track['proc_track']['y'][fidx]})
    vOut = None
    if render_video:
        firstImage = cv2.imread(flist[0])
        fw = firstImage.shape[1]
        fh = firstImage.shape[0]
        vOut = cv2.VideoWriter(os.path.join(args.pyaviPath, 'video_only.avi'), cv2.VideoWriter_fourcc(*'XVID'), 25, (fw,fh))
    colorDict = {0: 0, 1: 255}
    best_persons = []
    for fidx, fname in tqdm.tqdm(enumerate(flist), total = len(flist)):
        if render_video:
            image = cv2.imread(fname)
        best_face_idx = -1
        draw_face_idx = -1
        best_face_score = -100
        for i, face in enumerate(faces[fidx]):
            if face['score'] > best_face_score:
                draw_face_idx = best_face_idx
                best_face_idx = i
                best_face_score = face['score']
            else:
                draw_face_idx = i

            if draw_face_idx == -1:
                continue
            draw_face = faces[fidx][draw_face_idx]
            clr = colorDict[0]
            #txt = round(draw_face['score'], 1)
            if render_video:
                cv2.rectangle(image, (int(draw_face['x']-draw_face['s']), int(draw_face['y']-draw_face['s'])), (int(draw_face['x']+draw_face['s']), int(draw_face['y']+draw_face['s'])),(0,255,255),10)
                cv2.rectangle(image, (int(draw_face['x_']-draw_face['s_']), int(draw_face['y_']-draw_face['s_'])), (int(draw_face['x_']+draw_face['s_']), int(draw_face['y_']+draw_face['s_'])),(0,125,255),10)
            #cv2.putText(image,'%s'%(txt), (int(draw_face['x']-draw_face['s']), int(draw_face['y']-draw_face['s'])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,clr,255-clr),5)
        
        if best_face_idx != -1 and best_face_score >= asd_threshold:
            face = faces[fidx][best_face_idx]
            clr = colorDict[1]
            #txt = round(face['score'], 1)
            if render_video:
                cv2.rectangle(image, (int(face['x']-face['s']), int(face['y']-face['s'])), (int(face['x']+face['s']), int(face['y']+face['s'])),(0,255,255), 10)
                cv2.rectangle(image, (int(face['x_']-face['s_']), int(face['y_']-face['s_'])), (int(face['x_']+face['s_']), int(face['y_']+face['s_'])),(0,125,255),10)
            #cv2.putText(image,'%s'%(txt), (int(face['x']-face['s']), int(face['y']-face['s'])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,clr,255-clr),5)
        
            best_track_idx = faces[fidx][best_face_idx]['track']
            best_frame_idx = -1
            for idx, frame in enumerate(tracks[best_track_idx]['track']['frame'].tolist()):
                if frame == fidx:
                    best_frame_idx = idx
            best_persons.append(tracks[best_track_idx]['track']['person_bbox'][best_frame_idx])
        else:
            best_persons.append([0, 0, 0, 0])
        
        if render_video:
            write_path = os.path.join('detect', str('%06d' % fidx) + '.jpg')
            cv2.imwrite(write_path, image)
            vOut.write(image)
    if render_video:
        vOut.release()
        command = ("ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic" % \
            (os.path.join(args.pyaviPath, 'video_only.avi'), os.path.join(args.pyaviPath, 'audio.wav'), \
            args.nDataLoaderThread, os.path.join(args.pyaviPath,'video_out.avi'))) 
        subprocess.call(command, shell=True, stdout=None)

    print(best_persons)
    return best_persons


# Main function
def main(args, detector_model=None):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but GPU-only mode is enabled.")
    # This preprocesstion is modified based on this [repository](https://github.com/joonson/syncnet_python).
    # ```
    # .
    # ├── pyavi
    # │   ├── audio.wav (Audio from input video)
    # │   ├── video.avi (Copy of the input video)
    # │   ├── video_only.avi (Output video without audio)
    # │   └── video_out.avi  (Output video with audio)
    # ├── pycrop (The detected face videos and audios)
    # │   ├── 000000.avi
    # │   ├── 000000.wav
    # │   ├── 000001.avi
    # │   ├── 000001.wav
    # │   └── ...
    # ├── pyframes (All the video frames in this video)
    # │   ├── 000001.jpg
    # │   ├── 000002.jpg
    # │   └── ...	
    # └── pywork
    #     ├── faces.pckl (face detection result)
    #     ├── scene.pckl (scene detection result)
    #     ├── scores.pckl (ASD result)
    #     └── tracks.pckl (face tracking result)
    # ```

    # Initialization 

    if os.path.isfile(args.pretrainModel) == False: # Download the pretrained model
        
        Link = "1AbN9fCf9IexMxEKXLQY2KYBlb-IhSEea"
        cmd = "gdown --id %s -O %s"%(Link, args.pretrainModel)
        subprocess.call(cmd, shell=True, stdout=None)

    stage_times = {}
    pipeline_start = time.perf_counter()


    args.pyaviPath = os.path.join(args.savePath, 'pyavi')
    args.pyframesPath = os.path.join(args.savePath, 'pyframes')
    args.pyworkPath = os.path.join(args.savePath, 'pywork')
    args.pycropPath = os.path.join(args.savePath, 'pycrop')
    if os.path.exists(args.savePath):
        rmtree(args.savePath)
    os.makedirs(args.pyaviPath, exist_ok = True) # The path for the input video, input audio, output video
    os.makedirs(args.pyframesPath, exist_ok = True) # Save all the video frames
    os.makedirs(args.pyworkPath, exist_ok = True) # Save the results in this process by the pckl method
    os.makedirs(args.pycropPath, exist_ok = True) # Save the detected face clips (audio+video) in this process

    # Extract audio
    args.audioFilePath = os.path.join(args.pyaviPath, 'audio.wav')
    stage_start = time.perf_counter()
    command = ("ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic" % \
        (args.videoFilePath, args.nDataLoaderThread, args.audioFilePath))
    subprocess.call(command, shell=True, stdout=None)
    stage_times["extract_audio_s"] = time.perf_counter() - stage_start
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Extract the audio and save in %s \r\n" %(args.audioFilePath))

    # Extract the video frames
    stage_start = time.perf_counter()
    process_video(args.videoFilePath, args.pyframesPath, max_width=getattr(args, "max_frame_width", None))
    stage_times["extract_frames_s"] = time.perf_counter() - stage_start

    # Scene detection for the video frames
    stage_start = time.perf_counter()
    scene = scene_detect(args)
    stage_times["scene_detect_s"] = time.perf_counter() - stage_start
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scene detection and save in %s \r\n" %(args.pyworkPath))	
    
    # TODO: person detection with mmdetection 
    stage_start = time.perf_counter()
    persons = get_bbox_data(args, detector_model=detector_model)
    stage_times["person_detect_s"] = time.perf_counter() - stage_start
    
    # Face detection for the video frames
    stage_start = time.perf_counter()
    faces = inference_video(args, persons)
    stage_times["face_detect_s"] = time.perf_counter() - stage_start
    
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face detection and save in %s \r\n" %(args.pyworkPath))

    # Face tracking
    stage_start = time.perf_counter()
    allTracks, vidTracks = [], []
    for shot in scene:
        if shot[1].frame_num - shot[0].frame_num >= args.minTrack: # Discard the shot frames less than minTrack frames
            allTracks.extend(track_shot(args, faces[shot[0].frame_num:shot[1].frame_num])) # 'frames' to present this tracks' timestep, 'bbox' presents the location of the faces
    stage_times["face_track_s"] = time.perf_counter() - stage_start
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face track and detected %d tracks \r\n" %len(allTracks))
    print(allTracks)

    if len(allTracks) == 0:
        stage_times["total_s"] = time.perf_counter() - pipeline_start
        with open(os.path.join(args.pyworkPath, "stage_timing.json"), "w", encoding="utf-8") as f:
            json.dump(stage_times, f, ensure_ascii=False, indent=2)
        raise RuntimeError("No valid face tracks were found in this video.")
    
    # Face clips cropping
    stage_start = time.perf_counter()
    for ii, track in tqdm.tqdm(enumerate(allTracks), total = len(allTracks)):
        vidTracks.append(crop_video(args, track, os.path.join(args.pycropPath, '%05d'%ii)))
    stage_times["face_crop_s"] = time.perf_counter() - stage_start
    
    
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Face Crop and saved in %s tracks \r\n" %args.pycropPath)
    

    # Active Speaker Detection by TalkNet
    stage_start = time.perf_counter()
    files = glob.glob("%s/*.avi"%args.pycropPath)
    files.sort()
    if len(files) == 0:
        raise RuntimeError("No cropped face videos were generated for TalkNet.")
    scores = evaluate_network(files, args)
    stage_times["talknet_asd_s"] = time.perf_counter() - stage_start
    print(scores)
    
    sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + " Scores extracted and saved in %s \r\n" %args.pyworkPath)

    # Visualization, save the result as the new video	
    stage_start = time.perf_counter()
    best_persons = visualization(vidTracks, scores, args, render_video=getattr(args, "render_video", False))
    stage_times["visualization_s"] = time.perf_counter() - stage_start
    np.save(os.path.join(args.pyworkPath, 'best_persons.npy'), np.array(best_persons))
    stage_times["total_s"] = time.perf_counter() - pipeline_start
    with open(os.path.join(args.pyworkPath, "stage_timing.json"), "w", encoding="utf-8") as f:
        json.dump(stage_times, f, ensure_ascii=False, indent=2)
    print("Stage timing (s): {}".format(stage_times))
    return best_persons

def build_runtime_args(
    video_file_path,
    save_path,
    pretrain_model=None,
    config_file=None,
    checkpoint_file=None,
    speaker_threshold=DEFAULT_ACTIVE_SPEAKER_THRESHOLD,
    face_detector_weights=DEFAULT_FACE_DETECTOR_WEIGHTS,
    face_det_conf=0.8,
    face_det_nms=0.3,
    face_det_topk=5000,
    face_det_max_side=320,
    face_redetect_interval=2,
    face_track_person_iou=0.5,
    face_track_min_conf=0.85,
    min_person_side=150,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but GPU-only mode is enabled.")
    runtime_device = "cuda:0"
    return argparse.Namespace(
        pretrainModel=pretrain_model or os.path.join(THIS_DIR, "pretrain_TalkSet.model"),
        savePath=save_path,
        videoFilePath=video_file_path,
        device=runtime_device,
        nDataLoaderThread=10,
        det_batch_size=2,
        face_detector_weights=face_detector_weights,
        face_det_conf=float(face_det_conf),
        face_det_nms=float(face_det_nms),
        face_det_topk=int(face_det_topk),
        face_det_max_side=int(face_det_max_side),
        face_redetect_interval=int(face_redetect_interval),
        face_track_person_iou=float(face_track_person_iou),
        face_track_min_conf=float(face_track_min_conf),
        min_person_side=int(min_person_side),
        minTrack=10,
        numFailedDet=10,
        log_every_frames=20,
        max_persons_per_frame=5,
        minFaceSize=1,
        cropScale=0.40,
        max_frame_width=None,
        render_video=False,
        evalCol=False,
        start=0,
        duration=0,
        colSavePath="/data08/col",
        activeSpeakerThreshold=float(speaker_threshold),
        config_file=config_file or DEFAULT_MMDET_CONFIG,
        checkpoint_file=checkpoint_file or DEFAULT_MMDET_CHECKPOINT,
    )


def run_single_video(
    video_file_path,
    save_path,
    pretrain_model=None,
    config_file=None,
    checkpoint_file=None,
    detector_model=None,
    speaker_threshold=DEFAULT_ACTIVE_SPEAKER_THRESHOLD,
    face_detector_weights=DEFAULT_FACE_DETECTOR_WEIGHTS,
    face_det_conf=0.8,
    face_det_nms=0.3,
    face_det_topk=5000,
    face_det_max_side=320,
    face_redetect_interval=2,
    face_track_person_iou=0.5,
    face_track_min_conf=0.85,
    min_person_side=150,
):
    args = build_runtime_args(
        video_file_path=video_file_path,
        save_path=save_path,
        pretrain_model=pretrain_model,
        config_file=config_file,
        checkpoint_file=checkpoint_file,
        speaker_threshold=speaker_threshold,
        face_detector_weights=face_detector_weights,
        face_det_conf=face_det_conf,
        face_det_nms=face_det_nms,
        face_det_topk=face_det_topk,
        face_det_max_side=face_det_max_side,
        face_redetect_interval=face_redetect_interval,
        face_track_person_iou=face_track_person_iou,
        face_track_min_conf=face_track_min_conf,
        min_person_side=min_person_side,
    )
    return main(args, detector_model=detector_model)


def detect_speaker_main(
    video_file_path=None,
    save_path=None,
    pretrain_model=None,
    config_file=None,
    checkpoint_file=None,
    speaker_threshold=DEFAULT_ACTIVE_SPEAKER_THRESHOLD,
    **kwargs
):
    """
    Backward-compatible entry point.
    Legacy callers may pass `model=...`; it will be treated as shared detector_model.
    """
    if video_file_path is None:
        video_file_path = os.path.join(PROJECT_ROOT, "raw_video", "single_input.mp4")

    if save_path is None:
        save_path = os.path.join(PROJECT_ROOT, "preprocessed_output", "detect_speaker", "single_clip")

    detector_model = kwargs.get("detector_model", kwargs.get("model"))
    face_detector_weights = kwargs.get("face_detector_weights", DEFAULT_FACE_DETECTOR_WEIGHTS)
    face_det_conf = kwargs.get("face_det_conf", 0.8)
    face_det_nms = kwargs.get("face_det_nms", 0.3)
    face_det_topk = kwargs.get("face_det_topk", 5000)
    face_det_max_side = kwargs.get("face_det_max_side", 320)
    face_redetect_interval = kwargs.get("face_redetect_interval", 2)
    face_track_person_iou = kwargs.get("face_track_person_iou", 0.5)
    face_track_min_conf = kwargs.get("face_track_min_conf", 0.85)
    min_person_side = kwargs.get("min_person_side", 150)

    return run_single_video(
        video_file_path=video_file_path,
        save_path=save_path,
        pretrain_model=pretrain_model,
        config_file=config_file,
        checkpoint_file=checkpoint_file,
        detector_model=detector_model,
        speaker_threshold=speaker_threshold,
        face_detector_weights=face_detector_weights,
        face_det_conf=face_det_conf,
        face_det_nms=face_det_nms,
        face_det_topk=face_det_topk,
        face_det_max_side=face_det_max_side,
        face_redetect_interval=face_redetect_interval,
        face_track_person_iou=face_track_person_iou,
        face_track_min_conf=face_track_min_conf,
        min_person_side=min_person_side,
    )


def _build_cli_parser():
    parser = argparse.ArgumentParser(description="Single-video TalkNet active speaker detection")
    parser.add_argument(
        "--videoFilePath",
        type=str,
        required=True,
        help="Input .mp4 path for one video clip.",
    )
    parser.add_argument(
        "--savePath",
        type=str,
        required=True,
        help="Output directory for one clip (contains pyavi/pyframes/pycrop/pywork).",
    )
    parser.add_argument(
        "--pretrainModel",
        type=str,
        default=os.path.join(THIS_DIR, "pretrain_TalkSet.model"),
        help="Path for the pretrained TalkNet model",
    )
    parser.add_argument("--nDataLoaderThread", type=int, default=10, help="Number of workers")
    parser.add_argument(
        "--det_batch_size",
        type=int,
        default=2,
        help="Batch size for Faster R-CNN person detection.",
    )
    parser.add_argument(
        "--log_every_frames",
        type=int,
        default=20,
        help="Print frame progress every N frames.",
    )
    parser.add_argument(
        "--max_persons_per_frame",
        type=int,
        default=5,
        help="Run face detection on top-K person boxes per frame (0 = all).",
    )
    parser.add_argument(
        "--face_detector_weights",
        type=str,
        default=DEFAULT_FACE_DETECTOR_WEIGHTS,
        help="Weights path for YuNet face detector backend.",
    )
    parser.add_argument(
        "--face_det_conf",
        type=float,
        default=0.8,
        help="Confidence threshold for face detector.",
    )
    parser.add_argument(
        "--face_det_nms",
        type=float,
        default=0.3,
        help="NMS threshold for YuNet face detector.",
    )
    parser.add_argument(
        "--face_det_topk",
        type=int,
        default=5000,
        help="Top-K faces kept by YuNet detector before NMS.",
    )
    parser.add_argument(
        "--face_det_max_side",
        type=int,
        default=320,
        help="Resize person crop longest side to this value before YuNet face detection (0 = no resize).",
    )
    parser.add_argument(
        "--face_redetect_interval",
        type=int,
        default=2,
        help="Run YuNet every N frames for each person; intermediate frames try box propagation.",
    )
    parser.add_argument(
        "--face_track_person_iou",
        type=float,
        default=0.5,
        help="Minimum person-box IoU to reuse previous frame face box.",
    )
    parser.add_argument(
        "--face_track_min_conf",
        type=float,
        default=0.85,
        help="Minimum previous face confidence required for propagation.",
    )
    parser.add_argument(
        "--min_person_side",
        type=int,
        default=150,
        help="Skip face detection for person crops smaller than this side length.",
    )
    parser.add_argument(
        "--minTrack", type=int, default=10, help="Number of min frames for each shot"
    )
    parser.add_argument(
        "--numFailedDet",
        type=int,
        default=10,
        help="Number of missed detections allowed before tracking stops",
    )
    parser.add_argument("--minFaceSize", type=int, default=1, help="Minimum face size in pixels")
    parser.add_argument("--cropScale", type=float, default=0.40, help="Scale bounding box")
    parser.add_argument(
        "--activeSpeakerThreshold",
        type=float,
        default=DEFAULT_ACTIVE_SPEAKER_THRESHOLD,
        help="Speaking probability threshold for selecting active speaker bbox per frame.",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=DEFAULT_MMDET_CONFIG,
        help="Detection config file.",
    )
    parser.add_argument(
        "--checkpoint_file",
        type=str,
        default=DEFAULT_MMDET_CHECKPOINT,
        help="Detection checkpoint file.",
    )
    return parser


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = _build_cli_parser()
    args = parser.parse_args()
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.evalCol = False
    args.start = 0
    args.duration = 0
    args.colSavePath = "/data08/col"
    print(main(args))
