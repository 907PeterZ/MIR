import os
import pickle
import gc
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import torch
from transformers import BertTokenizer

from backbones.base import ModelManager

# ================== 配置区 ==================
ROOT = os.path.dirname(os.path.abspath(__file__))
LOCAL_BERT_PATH = os.path.join(ROOT, "cache", "local_bert_tokenizer")
MODEL_WEIGHTS_MAP = {
    "mult": os.path.join(ROOT, "cache", "MULT", "pytorch_model.bin"),
}
FUSION_METHOD = "mult"
if FUSION_METHOD not in MODEL_WEIGHTS_MAP:
    raise ValueError("Unsupported fusion method: {}. Choose from {}".format(FUSION_METHOD, sorted(MODEL_WEIGHTS_MAP)))
MODEL_WEIGHTS_PATH = MODEL_WEIGHTS_MAP[FUSION_METHOD]
GPU_ID = 0

# ================== MIntRec 配置 ==================
benchmarks = {
    'MIntRec':{
        'intent_labels': [
            'Complain', 'Praise', 'Apologise', 'Thank', 'Criticize',
            'Agree', 'Taunt', 'Flaunt',
            'Joke', 'Oppose',
            'Comfort', 'Care', 'Inform', 'Advise', 'Arrange', 'Introduce',
            'Leave', 'Prevent', 'Greet', 'Ask for help'
        ],
        'binary_maps': {
            'Complain': 'Emotion', 'Praise':'Emotion', 'Apologise': 'Emotion',
            'Thank':'Emotion', 'Criticize': 'Emotion', 'Care': 'Emotion',
            'Agree': 'Emotion', 'Taunt': 'Emotion', 'Flaunt': 'Emotion',
            'Joke':'Emotion', 'Oppose': 'Emotion',
            'Inform':'Goal', 'Advise':'Goal', 'Arrange': 'Goal',
            'Introduce': 'Goal', 'Leave':'Goal', 'Prevent':'Goal',
            'Greet': 'Goal', 'Ask for help': 'Goal', 'Comfort': 'Goal'
        },
        'max_seq_lengths':{
            'text': 30,
            'video': 230,
            'audio': 480,
        },
        'feat_dims':{
            'text': 768,
            'video': 256,
            'audio': 768
        }
    }
}

INTENT_LABELS = benchmarks['MIntRec']['intent_labels']
BINARY_MAPS = benchmarks['MIntRec']['binary_maps']
MAX_SEQ = benchmarks['MIntRec']['max_seq_lengths']
FEAT_DIMS = benchmarks['MIntRec']['feat_dims']
NUM_LABELS = len(INTENT_LABELS)

# =================================================
def load_pkl(path, expected_dim=None):
    """
    加载并修复常见 pkl 特征格式
    返回: torch.Tensor [T, D]
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        # 取第一个 key（适配你现在生成的 pkl）
        data = next(iter(data.values()))

    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    else:
        data = np.array(data)

    if data.ndim == 3:
        data = data.squeeze()

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if expected_dim and data.shape[1] != expected_dim:
        print(f"⚠️ Warning: expect {expected_dim}, got {data.shape[1]}")

    return torch.tensor(data, dtype=torch.float32)

# =================================================
def _build_inference_args():
    common_args = dict(
        method=FUSION_METHOD,
        logger_name="Inference",
        gpu_id=GPU_ID,
        pretrained_model_path=MODEL_WEIGHTS_PATH,
        text_backbone=LOCAL_BERT_PATH,
        cache_path=LOCAL_BERT_PATH,
        num_labels=NUM_LABELS,
        text_seq_len=MAX_SEQ["text"],
        audio_seq_len=MAX_SEQ["audio"],
        video_seq_len=MAX_SEQ["video"],
        text_feat_dim=FEAT_DIMS["text"],
        audio_feat_dim=FEAT_DIMS["audio"],
        video_feat_dim=FEAT_DIMS["video"],
        use_cuda=torch.cuda.is_available(),
    )

    if FUSION_METHOD == "mult":
        # Match MULT config defaults for checkpoint compatibility.
        common_args.update(
            need_aligned=False,
            dst_feature_dims=120,
            nheads=8,
            n_levels=8,
            attn_dropout=0.0,
            attn_dropout_v=0.2,
            attn_dropout_a=0.2,
            relu_dropout=0.0,
            embed_dropout=0.1,
            res_dropout=0.0,
            output_dropout=0.2,
            text_dropout=0.4,
            attn_mask=True,
            conv1d_kernel_size_l=5,
            conv1d_kernel_size_v=1,
            conv1d_kernel_size_a=1,
        )
    else:
        common_args.update(
            need_aligned=True,
            aligned_method="ctc",
            beta_shift=0.005,
            dropout_prob=0.5,
        )

    args = SimpleNamespace(**common_args)
    args.device = torch.device("cuda:{}".format(GPU_ID) if args.use_cuda else "cpu")
    return args


@lru_cache(maxsize=1)
def _get_cached_model_bundle():
    if not os.path.exists(LOCAL_BERT_PATH):
        raise FileNotFoundError("Tokenizer path not found: {}".format(LOCAL_BERT_PATH))
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        raise FileNotFoundError("Model checkpoint not found: {}".format(MODEL_WEIGHTS_PATH))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, but GPU-only mode is enabled.")

    args = _build_inference_args()
    model_manager = ModelManager(args)
    model = model_manager.model.to(args.device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(LOCAL_BERT_PATH)
    return model, tokenizer, args.device


def load_inference_model():
    """
    返回缓存后的 (model, tokenizer, device)，可复用避免重复加载权重。
    """
    return _get_cached_model_bundle()


def clear_inference_model_cache():
    """
    Clear cached MULT model/tokenizer to release memory on explicit reset.
    """
    _get_cached_model_bundle.cache_clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    gc.collect()

# =================================================
def run_inference(
    text_input,
    video_pkl,
    audio_pkl,
    model_bundle=None,
):
    """
    完整推理流程：
    默认使用缓存模型，避免每次推理都重新加载权重。
    """
    if model_bundle is None:
        model, tokenizer, device = load_inference_model()
    else:
        model, tokenizer, device = model_bundle

    # ========= 文本处理 =========
    inputs = tokenizer(
        text_input,
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ["text"],
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    token_type_ids = torch.zeros_like(input_ids)

    text_feats = torch.stack(
        [input_ids, attention_mask, token_type_ids], dim=1
    ).to(device)

    # ========= 加载特征 =========
    video_feats = load_pkl(
        video_pkl,
        FEAT_DIMS["video"]
    ).unsqueeze(0).to(device)

    audio_feats = load_pkl(
        audio_pkl,
        FEAT_DIMS["audio"]
    ).unsqueeze(0).to(device)

    # ========= 推理 =========
    with torch.no_grad():
        model_output = model(text_feats, video_feats, audio_feats)

    if isinstance(model_output, (tuple, list)):
        logits = model_output[0]
    else:
        logits = model_output

    probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
    pred_idx = int(np.argmax(probs))

    return {
        "intent": INTENT_LABELS[pred_idx],
        "binary": BINARY_MAPS.get(INTENT_LABELS[pred_idx], "Unknown"),
        "probs": probs.tolist()
    }

# =================================================
if __name__ == "__main__":
    # CLI 测试用

    result = run_inference(
        text_input="for Amy's Goodbye party. And I was hoping that you could come up with some electronic photos every time here. J-j-j-j.",
        video_pkl="./preprocessed_output/video_feats.pkl",
        audio_pkl="./preprocessed_output/audio_feats.pkl"
    )

    print("Intent:", result["intent"])
    print("Binary:", result["binary"])
