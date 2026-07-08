"""
Knowledge distillation training script.

Decomposes a pretrained model using BTT/SVD methods, then fine-tunes
with knowledge distillation loss against precomputed teacher data.

Examples:
  # Decompose with C4, SFT KD
  python src/run_kd.py --config recipes/kd/config_default.yaml \
      --model_name_or_path Qwen/Qwen2.5-0.5B \
      --train_mode svd_llm \
      --kd_loss_type sft \
      --teacher_data_dir /data/yequan/fura/kd_data/DeepSeek-R1-Distill-Qwen-7B-competition_math

  # Decompose with traces, offline KL KD
  python src/run_kd.py --config recipes/kd/config_default.yaml \
      --model_name_or_path Qwen/Qwen2.5-0.5B \
      --train_mode svd_llm_v2 \
      --calib_source math_traces \
      --calib_traces_path outputs/math_traces/DeepSeek-R1-Distill-Qwen-1.5B/traces.jsonl \
      --kd_loss_type kl \
      --teacher_data_dir /data/yequan/fura/kd_data/DeepSeek-R1-Distill-Qwen-7B-competition_math
"""

import json
import math
import os
import pathlib
import sys
import time
import warnings
from dataclasses import dataclass, field
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

# Heavy training deps (torch, transformers) are imported at module level.
# wandb and safetensors are wrapped in try/except for optional use.
try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]

try:
    from safetensors.torch import load_file as load_safetensors_file
    from safetensors.torch import save_file as save_safetensors_file
except ImportError:
    load_safetensors_file = None  # type: ignore[assignment]
    save_safetensors_file = None  # type: ignore[assignment]

# Make repo packages importable.
# compress_then_train.py lives in src/, so repo_root = src/../ = project root.
# Adding src/ lets us import the `compress` submodule; adding the repo root lets
# us import the top-level `eval` package (lm-eval-harness adaptor).
_src_dir = pathlib.Path(__file__).resolve().parent          # src/
_repo_root = _src_dir.parent                                # project root
for _p in (str(_src_dir), str(_repo_root)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from compress.ppl_eval import evaluate_model_ppl  # noqa: E402

_VALID_TRAIN_MODES = {
    "full",
    "svd",
    "svd_llm",
    "svd_llm_v2",
    "svd_llm_v2_bp",
    "svd_llm_v2_combined",
    "svd_als",
    "svd_twosteps",
    "btt",
    "btt_llm_v2",
    "btt_llm_v2_bp",
    "btt_llm_v2_combined",
    "btt_twosteps",
}
_CALIB_FREE_TRAIN_MODES = {"svd", "btt"}  # do not need calibration data
_VALID_CALIB_SOURCES = {"c4", "traces", "training_data"}
_VALID_KD_LOSS_TYPES = {"sft", "kl", "kl_online", "ce"}
_VALID_OPTIMIZERS = {"adamw", "adamw_8bit"}
_VALID_SCHEDULERS = {"none", "linear", "cosine"}
_BTT_TRAIN_MODES = {"btt", "btt_llm_v2", "btt_llm_v2_bp", "btt_llm_v2_combined", "btt_twosteps"}
_VALID_BTT_DECOMP_MODES = {"square", "input_one_block", "output_one_block"}
_VALID_BTT_TRAIN_POSITIONS = {"small", "large", "both"}


@dataclass
class KDScriptArguments:
    """Model loading and training hyperparameters."""

    model_name_or_path: str = field(
        metadata={"help": "Student model HF id or local path."}
    )
    lr: float = field(default=1e-4, metadata={"help": "Learning rate."})
    optimizer: str = field(
        default="adamw",
        metadata={"help": "Optimizer. Choices: adamw, adamw_8bit.", "choices": ["adamw", "adamw_8bit"]},
    )
    lr_scheduler: str = field(
        default="cosine",
        metadata={"help": "LR scheduler. Choices: none, linear, cosine.", "choices": ["none", "linear", "cosine"]},
    )
    warmup_ratio: float = field(default=0.05, metadata={"help": "Warmup fraction of total optimizer steps."})
    weight_decay: float = field(default=0.01, metadata={"help": "AdamW weight decay."})
    batch_size: int = field(default=2, metadata={"help": "Per-device training batch size."})
    gradient_accumulation_steps: int = field(default=16, metadata={"help": "Gradient accumulation steps."})
    num_epochs: int = field(default=1, metadata={"help": "Number of training epochs."})
    seed: int = field(default=42, metadata={"help": "Random seed."})
    wandb_project: Optional[str] = field(default=None, metadata={"help": "W&B project name."})
    wandb_run_name: Optional[str] = field(default=None, metadata={"help": "W&B run name."})
    name_suffix: Optional[str] = field(
        default=None,
        metadata={"help": "Optional suffix appended to the generated run name."},
    )
    no_wandb: bool = field(default=False, metadata={"help": "Disable W&B logging."})

    def __post_init__(self):
        if self.optimizer not in _VALID_OPTIMIZERS:
            raise ValueError(
                f"optimizer must be one of {sorted(_VALID_OPTIMIZERS)}, got {self.optimizer!r}"
            )
        if self.lr_scheduler not in _VALID_SCHEDULERS:
            raise ValueError(
                f"lr_scheduler must be one of {sorted(_VALID_SCHEDULERS)}, got {self.lr_scheduler!r}"
            )
        if not (0.0 <= self.warmup_ratio <= 1.0):
            raise ValueError(f"warmup_ratio must be in [0, 1], got {self.warmup_ratio}")


@dataclass
class KDDecompositionConfig:
    """Controls the decomposition step applied before KD training."""

    train_mode: str = field(
        default="full",
        metadata={
            "help": "Decomposition method. 'full' skips decomposition.",
            "choices": [
                "full",
                "svd",
                "svd_llm",
                "svd_llm_v2",
                "svd_llm_v2_bp",
                "svd_llm_v2_combined",
                "svd_als",
                "svd_twosteps",
                "btt",
                "btt_llm_v2",
                "btt_llm_v2_bp",
                "btt_llm_v2_combined",
                "btt_twosteps",
            ],
        },
    )
    compression_ratio: float = field(
        default=0.7,
        metadata={"help": "Fraction of params to retain (e.g. 0.7 = 70%). Ignored when train_mode=full."},
    )
    calib_source: str = field(
        default="c4",
        metadata={
            "help": (
                "Calibration data source for decomposition. "
                "c4: stream from HuggingFace C4. "
                "traces: local JSONL at calib_traces_path. "
                "training_data: use the KD training completions from teacher_data_dir."
            ),
            "choices": ["c4", "traces", "training_data"],
        },
    )
    calib_traces_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to trace JSONL. Required when calib_source=traces."},
    )
    calib_num_seqs: int = field(default=128, metadata={"help": "Number of calibration sequences."})
    calib_max_length: int = field(default=2048, metadata={"help": "Max token length per calibration sequence."})
    calib_seed: int = field(default=3, metadata={"help": "Random seed for SVD-LLM-compatible calibration sampling."})
    eval_ppl_after_compression: bool = field(
        default=True,
        metadata={"help": "Run WikiText-2/C4 perplexity evaluation immediately after decomposition."},
    )
    eval_ppl_seqlen: int = field(
        default=2048,
        metadata={"help": "Sequence length for post-decomposition PPL evaluation."},
    )
    eval_ppl_seed: int = field(
        default=0,
        metadata={"help": "Random seed for post-decomposition PPL evaluation."},
    )
    skip_layers: str = field(
        default="lm_head",
        metadata={"help": "Comma-separated leaf layer names to skip during decomposition."},
    )
    save_decomposed_dir: Optional[str] = field(
        default=None,
        metadata={"help": "If set, save the compressed model here before KD training."},
    )
    als_n_iter: int = field(
        default=10,
        metadata={"help": "Max ALS iterations per layer (svd_als only)."},
    )
    als_tol: float = field(
        default=1e-6,
        metadata={"help": "ALS early stopping relative tolerance (svd_als only)."},
    )
    als_weighting: str = field(
        default="equal",
        metadata={
            "help": "ALS alpha/beta weighting mode (svd_als only).",
            "choices": ["equal", "trace"],
        },
    )
    als_reg_eps: float = field(
        default=1e-4,
        metadata={"help": "Regularization epsilon for ALS eigendecomposition (svd_als only)."},
    )
    twosteps_n_refine: int = field(
        default=1,
        metadata={"help": "Refinement iterations for svd_twosteps."},
    )
    twosteps_reg_eps: float = field(
        default=1e-4,
        metadata={"help": "Regularization epsilon for svd_twosteps solves/whitening."},
    )
    decomp_mode: str = field(
        default="square",
        metadata={
            "help": "BTT decomposition mode (BTT train modes only).",
            "choices": ["square", "input_one_block", "output_one_block"],
        },
    )
    train_position: str = field(
        default="both",
        metadata={
            "help": "BTT trainable-side selector (BTT train modes only): small|large|both.",
            "choices": ["small", "large", "both"],
        },
    )

    def __post_init__(self):
        if self.train_mode not in _VALID_TRAIN_MODES:
            raise ValueError(
                f"train_mode must be one of {sorted(_VALID_TRAIN_MODES)}, got {self.train_mode!r}"
            )
        if self.calib_source not in _VALID_CALIB_SOURCES:
            raise ValueError(
                f"calib_source must be one of {sorted(_VALID_CALIB_SOURCES)}, got {self.calib_source!r}"
            )
        if self.train_mode in _CALIB_FREE_TRAIN_MODES:
            if self.calib_source != "c4":
                warnings.warn(
                    f"calib_source={self.calib_source!r} is ignored for calibration-free "
                    f"train_mode={self.train_mode!r}.",
                    stacklevel=2,
                )
            if self.calib_traces_path:
                warnings.warn(
                    f"calib_traces_path is ignored for calibration-free train_mode={self.train_mode!r}.",
                    stacklevel=2,
                )
        else:
            if self.calib_source == "traces" and not self.calib_traces_path:
                raise ValueError("calib_traces_path must be set when calib_source=traces")
        if self.twosteps_n_refine < 0:
            raise ValueError(f"twosteps_n_refine must be >= 0, got {self.twosteps_n_refine}")
        if self.twosteps_reg_eps <= 0:
            raise ValueError(f"twosteps_reg_eps must be > 0, got {self.twosteps_reg_eps}")
        if self.train_mode in _BTT_TRAIN_MODES:
            if self.decomp_mode not in _VALID_BTT_DECOMP_MODES:
                raise ValueError(
                    f"decomp_mode must be one of {sorted(_VALID_BTT_DECOMP_MODES)} for BTT train modes, "
                    f"got {self.decomp_mode!r}"
                )
            if self.train_position not in _VALID_BTT_TRAIN_POSITIONS:
                raise ValueError(
                    f"train_position must be one of {sorted(_VALID_BTT_TRAIN_POSITIONS)} for BTT train modes, "
                    f"got {self.train_position!r}"
                )


@dataclass
class KDTrainingConfig:
    """Controls the KD loss type and teacher data."""

    kd_loss_type: str = field(
        metadata={
            "help": (
                "KD loss type: sft (CE on teacher completions), kl (offline top-K KL), "
                "kl_online (live KL), ce (next-token CE on general text, no teacher needed)."
            ),
            "choices": ["sft", "kl", "kl_online", "ce"],
        }
    )
    teacher_data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to dir with completions.jsonl and logits/ chunks. Not used when kd_loss_type=ce."},
    )
    top_k: int = field(default=256, metadata={"help": "Top-K logits for KL loss; must match generated data."})
    teacher_model_id: Optional[str] = field(
        default=None,
        metadata={"help": "Teacher model HF id for kl_online mode."},
    )
    max_length: int = field(default=2048, metadata={"help": "Max token length for KD sequences."})
    ce_seq_len: int = field(default=256, metadata={"help": "Fixed sequence length for CE training samples."})
    ce_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Total optimizer update steps for CE mode. Required when kd_loss_type=ce."},
    )
    ce_data_source: str = field(
        default="c4",
        metadata={
            "help": "Text corpus for ce mode. Choices: c4 (streamed HF), jsonl (local file with 'text' field).",
            "choices": ["c4", "jsonl"],
        },
    )
    ce_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to JSONL with 'text' field. Required when kd_loss_type=ce and ce_data_source=jsonl."},
    )
    ce_num_seqs: int = field(
        default=2048,
        metadata={"help": "Deprecated in CE mode; retained for compatibility with older configs."},
    )
    save_steps: str = field(
        default="10,30,final",
        metadata={"help": "Comma-separated optimizer steps to save checkpoints (e.g. '10,30,final') or 'none'."},
    )
    base_dir: str = field(default="outputs/kd_runs", metadata={"help": "Base directory for run outputs."})
    run_lm_eval: bool = field(
        default=True,
        metadata={"help": "Run lm-eval-harness benchmarks (before compression and after training)."},
    )
    lm_eval_tasks: str = field(
        default="hellaswag,mmlu",
        metadata={"help": "Comma-separated lm-eval-harness task names to benchmark."},
    )
    lm_eval_limit: int = field(
        default=-1,
        metadata={"help": "Per-task sample cap for lm-eval (<=0 means the full task)."},
    )
    lm_eval_batch_size: int = field(
        default=4,
        metadata={"help": "Batch size for lm-eval-harness evaluation."},
    )
    lm_eval_max_seqlen: int = field(
        default=2048,
        metadata={"help": "Max sequence length for lm-eval-harness evaluation."},
    )
    eval_before_compression: bool = field(
        default=True,
        metadata={"help": "Also benchmark the uncompressed base model for a before/after comparison."},
    )

    def __post_init__(self):
        if self.kd_loss_type not in _VALID_KD_LOSS_TYPES:
            raise ValueError(
                f"kd_loss_type must be one of {sorted(_VALID_KD_LOSS_TYPES)}, got {self.kd_loss_type!r}"
            )
        if self.kd_loss_type in {"sft", "kl", "kl_online"} and not self.teacher_data_dir:
            raise ValueError("teacher_data_dir must be set when kd_loss_type is sft/kl/kl_online")
        if self.kd_loss_type == "kl_online" and not self.teacher_model_id:
            raise ValueError("teacher_model_id must be set when kd_loss_type=kl_online")
        if self.ce_seq_len <= 1:
            raise ValueError(f"ce_seq_len must be > 1, got {self.ce_seq_len}")
        if self.kd_loss_type == "ce" and (self.ce_steps is None or self.ce_steps <= 0):
            raise ValueError("ce_steps must be set to a positive integer when kd_loss_type=ce")
        if self.kd_loss_type == "ce" and self.ce_data_source == "jsonl" and not self.ce_data_path:
            raise ValueError("ce_data_path must be set when kd_loss_type=ce and ce_data_source=jsonl")


# ══════════════════════════════════════════════════════════════════════════════
#  KD Datasets
# ══════════════════════════════════════════════════════════════════════════════

class KDSftDataset(Dataset):
    """SFT-style KD: train the student to reproduce teacher completions."""

    def __init__(self, completions, max_length=2048):
        self.completions = completions
        self.max_length = max_length

    def __len__(self):
        return len(self.completions)

    def __getitem__(self, idx):
        entry = self.completions[idx]
        completion_ids = entry["token_ids"]
        input_ids = completion_ids[: self.max_length]
        labels = input_ids.copy()
        attention_mask = [1] * len(input_ids)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


class KDKlDataset(Dataset):
    """Offline KL KD: match top-K teacher logits from precomputed chunks."""

    def __init__(self, completions, teacher_data_dir, top_k, max_length=2048, index_offset=0):
        self.completions = completions
        self.top_k = top_k
        self.max_length = max_length
        self.index_offset = index_offset

        logits_dir = os.path.join(teacher_data_dir, "logits")
        chunk_files = sorted(
            f for f in os.listdir(logits_dir) if f.startswith("chunk_") and f.endswith(".safetensors")
        )
        self.all_topk_values = []
        self.all_topk_indices = []
        self.all_seq_lengths = []
        for chunk_file in chunk_files:
            chunk = load_safetensors_file(os.path.join(logits_dir, chunk_file))
            n = chunk["seq_lengths"].shape[0]
            for i in range(n):
                self.all_topk_values.append(chunk["topk_values"][i])
                self.all_topk_indices.append(chunk["topk_indices"][i])
                self.all_seq_lengths.append(chunk["seq_lengths"][i].item())

    def __len__(self):
        return len(self.completions)

    def __getitem__(self, idx):
        entry = self.completions[idx]
        completion_ids = entry["token_ids"]
        logit_idx = idx + self.index_offset
        input_ids = completion_ids[: self.max_length]
        actual_len = len(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * actual_len,
            "response_mask": [1] * actual_len,
            "teacher_topk_values": self.all_topk_values[logit_idx][:actual_len],
            "teacher_topk_indices": self.all_topk_indices[logit_idx][:actual_len],
        }


class KDOnlineDataset(Dataset):
    """Online KL KD: feed token sequences to both student and live teacher."""

    def __init__(self, completions, max_length=2048):
        self.completions = completions
        self.max_length = max_length

    def __len__(self):
        return len(self.completions)

    def __getitem__(self, idx):
        entry = self.completions[idx]
        input_ids = entry["token_ids"][: self.max_length]
        attention_mask = [1] * len(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "response_mask": attention_mask.copy()}


class KDCeDataset(Dataset):
    """CE training on raw token sequences — no teacher data required."""

    def __init__(self, sequences, max_length=2048):
        self.sequences = sequences
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_ids = self.sequences[idx][: self.max_length]
        return {"input_ids": input_ids, "labels": input_ids.copy(), "attention_mask": [1] * len(input_ids)}


# ══════════════════════════════════════════════════════════════════════════════
#  Loss functions
# ══════════════════════════════════════════════════════════════════════════════

def compute_kl_loss(student_logits, teacher_topk_values, teacher_topk_indices, response_mask):
    """KL(student || teacher) over teacher's top-K positions.

    Args:
        student_logits: [B, T, vocab_size]
        teacher_topk_values: [B, T, K] — teacher log-probs from vLLM
        teacher_topk_indices: [B, T, K] — token indices for top-K
        response_mask: [B, T]

    Returns:
        Scalar KL loss averaged over response tokens.
    """
    vocab_size = student_logits.shape[2]
    oob_mask = teacher_topk_indices >= vocab_size
    if oob_mask.any():
        teacher_topk_indices = teacher_topk_indices.clone()
        teacher_topk_indices[oob_mask] = 0

    student_log_probs_full = F.log_softmax(student_logits.float(), dim=-1)  # [B, T, V]
    student_log_probs = torch.gather(student_log_probs_full, dim=2, index=teacher_topk_indices.long())  # [B, T, K]
    teacher_log_probs = F.log_softmax(teacher_topk_values.float(), dim=-1)  # [B, T, K]

    kl_per_k = F.kl_div(teacher_log_probs, student_log_probs, log_target=True, reduction="none")
    if oob_mask.any():
        kl_per_k = kl_per_k.masked_fill(oob_mask, 0.0)

    kl_per_token = kl_per_k.sum(dim=-1)
    masked_kl = kl_per_token * response_mask
    return masked_kl.sum() / response_mask.sum().clamp(min=1)


def compute_online_kl_loss(student_logits, teacher_logits, response_mask, shared_vocab_size=None):
    """KL(student || teacher) over the full shared vocab.

    Args:
        student_logits: [B, T, student_vocab_size]
        teacher_logits: [B, T, teacher_vocab_size]
        response_mask: [B, T]
        shared_vocab_size: if set, both tensors are sliced to this size

    Returns:
        Scalar KL loss averaged over response tokens.
    """
    if shared_vocab_size is not None:
        student_logits = student_logits[:, :, :shared_vocab_size]
        teacher_logits = teacher_logits[:, :, :shared_vocab_size]

    teacher_log_probs = F.log_softmax(teacher_logits.float(), dim=-1)
    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    kl_per_token = F.kl_div(
        teacher_log_probs, student_log_probs, log_target=True, reduction="none"
    ).sum(dim=-1)
    masked_kl = kl_per_token * response_mask
    return masked_kl.sum() / response_mask.sum().clamp(min=1)


# ══════════════════════════════════════════════════════════════════════════════
#  Collate functions
# ══════════════════════════════════════════════════════════════════════════════

def build_kd_sft_collate_fn(pad_token_id):
    def collate_fn(batch):
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids, labels, attention_mask = [], [], []
        for item in batch:
            pad = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_token_id] * pad)
            labels.append(item["labels"] + [-100] * pad)
            attention_mask.append(item["attention_mask"] + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention_mask),
        }
    return collate_fn


def build_kd_kl_collate_fn(pad_token_id, top_k):
    def collate_fn(batch):
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids, attention_mask, response_mask = [], [], []
        teacher_topk_values, teacher_topk_indices = [], []
        for item in batch:
            actual_len = len(item["input_ids"])
            pad = max_len - actual_len
            input_ids.append(item["input_ids"] + [pad_token_id] * pad)
            attention_mask.append(item["attention_mask"] + [0] * pad)
            response_mask.append(item["response_mask"] + [0] * pad)
            tv, ti = item["teacher_topk_values"], item["teacher_topk_indices"]
            if pad > 0:
                tv = torch.cat([tv, torch.zeros(pad, top_k, dtype=tv.dtype)])
                ti = torch.cat([ti, torch.zeros(pad, top_k, dtype=ti.dtype)])
            teacher_topk_values.append(tv)
            teacher_topk_indices.append(ti)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "response_mask": torch.tensor(response_mask, dtype=torch.float32),
            "teacher_topk_values": torch.stack(teacher_topk_values),
            "teacher_topk_indices": torch.stack(teacher_topk_indices),
        }
    return collate_fn


def build_kd_online_collate_fn(pad_token_id):
    def collate_fn(batch):
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids, attention_mask, response_mask = [], [], []
        for item in batch:
            pad = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_token_id] * pad)
            attention_mask.append(item["attention_mask"] + [0] * pad)
            response_mask.append(item["response_mask"] + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "response_mask": torch.tensor(response_mask, dtype=torch.float32),
        }
    return collate_fn


# ══════════════════════════════════════════════════════════════════════════════
#  Decomposition
# ══════════════════════════════════════════════════════════════════════════════

def _build_calib_loader(tokenizer, decomp_args: KDDecompositionConfig, train_completions=None):
    """Build a DataLoader for calibration.

    Supports three sources (decomp_args.calib_source):
      c4            — stream from HuggingFace C4
      traces        — local JSONL at calib_traces_path (prompt+completion text)
      training_data — use train_completions passed in from the KD training set

    Backed by the self-contained loaders in the `compress` package, which pack
    calibration text into fixed-length token windows (matching the covariance
    calibration used by the SVD/BTT decomposition methods).
    """
    from compress.loaders import (
        build_c4_calib_loader,
        build_text_calib_loader,
        build_traces_jsonl_calib_loader,
    )

    if decomp_args.calib_source == "c4":
        return build_c4_calib_loader(
            tokenizer,
            num_seqs=decomp_args.calib_num_seqs,
            max_length=decomp_args.calib_max_length,
            batch_size=8,
            seed=decomp_args.calib_seed,
        )
    if decomp_args.calib_source == "traces":
        return build_traces_jsonl_calib_loader(
            tokenizer,
            decomp_args.calib_traces_path,
            num_seqs=decomp_args.calib_num_seqs,
            max_length=decomp_args.calib_max_length,
            batch_size=8,
        )
    # training_data
    if not train_completions:
        raise ValueError(
            "calib_source='training_data' requires train_completions; "
            "ensure kd_loss_type is not 'ce' so training data is loaded."
        )
    texts = [c["prompt"] + c["completion"] for c in train_completions]
    return build_text_calib_loader(
        tokenizer,
        texts,
        num_seqs=decomp_args.calib_num_seqs,
        max_length=decomp_args.calib_max_length,
        batch_size=8,
    )


def _resolve_btt_trainable_sides(left_size: int, right_size: int, train_position: str) -> Tuple[bool, bool]:
    if train_position not in _VALID_BTT_TRAIN_POSITIONS:
        raise ValueError(
            f"train_position must be one of {sorted(_VALID_BTT_TRAIN_POSITIONS)}, got {train_position!r}"
        )
    if train_position == "both":
        return True, True
    if train_position == "small":
        train_left = left_size <= right_size
        return train_left, not train_left
    train_left = left_size >= right_size
    return train_left, not train_left


def configure_btt_trainability(
    model: torch.nn.Module,
    train_position: str = "both",
    train_bias: bool = True,
) -> dict:
    """Freeze all params, then unfreeze BTT cores according to train_position."""
    from compress.btt_linear import BTTLinear

    for p in model.parameters():
        p.requires_grad = False

    num_btt_layers = 0
    tuned_left_cores = 0
    tuned_right_cores = 0
    tuned_biases = 0

    for _, module in model.named_modules():
        if not isinstance(module, BTTLinear):
            continue
        num_btt_layers += 1
        left_size = module.btt_l.numel()
        right_size = module.btt_r.numel()
        train_left, train_right = _resolve_btt_trainable_sides(left_size, right_size, train_position)
        module.btt_l.requires_grad = train_left
        module.btt_r.requires_grad = train_right
        tuned_left_cores += int(train_left)
        tuned_right_cores += int(train_right)

        if module.bias is not None:
            module.bias.requires_grad = train_bias
            tuned_biases += int(train_bias)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_param_count = sum(p.numel() for p in trainable_params)
    total_param_count = sum(p.numel() for p in model.parameters())
    return {
        "num_btt_layers": num_btt_layers,
        "tuned_left_cores": tuned_left_cores,
        "tuned_right_cores": tuned_right_cores,
        "tuned_biases": tuned_biases,
        "trainable_param_count": trainable_param_count,
        "total_param_count": total_param_count,
        "trainable_params": trainable_params,
    }


def decompose_model(model, tokenizer, decomp_args: KDDecompositionConfig,
                    train_completions=None):
    """Decompose a model in-place using BTT/SVD methods.

    Returns the model unchanged when train_mode='full'.
    For BTT train modes, applies PEFT-style trainability to BTT cores only.
    For non-BTT modes, all parameters are made trainable.
    """
    if decomp_args.train_mode not in _VALID_TRAIN_MODES:
        raise ValueError(
            f"train_mode must be one of {sorted(_VALID_TRAIN_MODES)}, got {decomp_args.train_mode!r}"
        )

    if decomp_args.train_mode == "full":
        return model

    from compress.compress_model import compress_model_with_loader

    skip_layers = tuple(s.strip() for s in decomp_args.skip_layers.split(",") if s.strip())

    if decomp_args.train_mode in _CALIB_FREE_TRAIN_MODES:
        calib_loader = None
        print(f"[decompose] Calibration-free method '{decomp_args.train_mode}' — skipping data pass.")
    else:
        print(f"[decompose] Building calibration loader (source={decomp_args.calib_source}) ...")
        calib_loader = _build_calib_loader(tokenizer, decomp_args, train_completions=train_completions)

    print(f"[decompose] Compressing with method={decomp_args.train_mode}, "
          f"ratio={decomp_args.compression_ratio:.0%} ...")
    compress_model_with_loader(
        model,
        calib_loader,
        compression_ratio=decomp_args.compression_ratio,
        method=decomp_args.train_mode,
        device="cuda" if torch.cuda.is_available() else "cpu",
        skip_layers=skip_layers,
        als_n_iter=decomp_args.als_n_iter,
        als_tol=decomp_args.als_tol,
        als_weighting=decomp_args.als_weighting,
        als_reg_eps=decomp_args.als_reg_eps,
        twosteps_n_refine=decomp_args.twosteps_n_refine,
        twosteps_reg_eps=decomp_args.twosteps_reg_eps,
        btt_decomp_mode=decomp_args.decomp_mode,
    )

    if decomp_args.train_mode in _BTT_TRAIN_MODES:
        stats = configure_btt_trainability(
            model,
            train_position=decomp_args.train_position,
            train_bias=True,
        )
        if stats["num_btt_layers"] == 0:
            raise ValueError("No BTT layers found after decomposition for BTT train mode.")
        print("[decompose] Done. Applied BTT trainability controls:")
        print(f"  decomp_mode={decomp_args.decomp_mode}, train_position={decomp_args.train_position}")
        print(
            f"  trainable params: {stats['trainable_param_count']:,} / {stats['total_param_count']:,} "
            f"({100.0 * stats['trainable_param_count'] / max(1, stats['total_param_count']):.4f}%)"
        )
        print(
            f"  tuned cores: left={stats['tuned_left_cores']}, right={stats['tuned_right_cores']}, "
            f"biases={stats['tuned_biases']}"
        )
    else:
        model.requires_grad_(True)
        print("[decompose] Done. All parameters set trainable.")

    if decomp_args.save_decomposed_dir:
        from safetensors.torch import save_file as _save_file
        out_dir = decomp_args.save_decomposed_dir
        os.makedirs(out_dir, exist_ok=True)
        state_dict = {n: p.detach().cpu() for n, p in model.named_parameters()}
        _save_file(state_dict, os.path.join(out_dir, "model.safetensors"))
        model.config.save_pretrained(out_dir)
        print(f"[decompose] Saved decomposed model to {out_dir}")

    return model


# ══════════════════════════════════════════════════════════════════════════════
#  Training utilities
# ══════════════════════════════════════════════════════════════════════════════

def compute_num_training_steps(num_batches, num_epochs, gradient_accumulation_steps):
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be > 0")
    return num_epochs * (num_batches // gradient_accumulation_steps)


def parse_save_steps(save_steps_str, total_steps):
    if save_steps_str is None:
        return set()
    if save_steps_str.strip().lower() == "none":
        return set()

    steps = set()
    for part in save_steps_str.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part == "final":
            steps.add(total_steps)
        else:
            steps.add(int(part))
    return steps


def is_ckpt_saving_disabled(save_steps_str):
    return (save_steps_str is None) or (save_steps_str.strip().lower() == "none")


def build_optimizer(args: KDScriptArguments, params):
    if args.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "adamw_8bit":
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("bitsandbytes is required for adamw_8bit; pip install bitsandbytes")
        return bnb.optim.AdamW8bit(params, lr=args.lr, weight_decay=args.weight_decay)
    raise ValueError(f"Unsupported optimizer: {args.optimizer!r}")


def _cosine_schedule_with_warmup_lambda(current_step, *, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def build_lr_scheduler(args: KDScriptArguments, optimizer, num_training_steps):
    warmup_steps = int(math.ceil(num_training_steps * args.warmup_ratio))
    if args.lr_scheduler == "none":
        return None
    if num_training_steps <= 0:
        raise ValueError(f"num_training_steps must be > 0 when using a scheduler, got {num_training_steps}")
    if args.lr_scheduler == "linear":
        return transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
        )
    if args.lr_scheduler == "cosine":
        fn = partial(
            _cosine_schedule_with_warmup_lambda,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
        return LambdaLR(optimizer, fn)
    raise ValueError(f"Unsupported lr_scheduler: {args.lr_scheduler!r}")


def save_kd_checkpoint(model, run_dir, step):
    ckpt_dir = os.path.join(run_dir, f"step={step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    state_dict = {name: param.detach().cpu().contiguous() for name, param in model.named_parameters()}
    save_safetensors_file(state_dict, os.path.join(ckpt_dir, "model.safetensors"))
    print(f"Saved checkpoint to {ckpt_dir}")


def save_kd_gradients(model, run_dir, step):
    """Save current parameter gradients to run_dir/step=<step>/grads.safetensors."""
    grad_dir = os.path.join(run_dir, f"step={step}")
    os.makedirs(grad_dir, exist_ok=True)
    grad_state = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_state[name] = param.grad.detach().cpu().contiguous()
    if not grad_state:
        print(f"Warning: no gradients found at step {step}; skipping gradient save.")
        return
    save_safetensors_file(grad_state, os.path.join(grad_dir, "grads.safetensors"))
    print(f"Saved gradients to {grad_dir}")


def maybe_save_gradient_snapshots(model, run_dir, save_steps, disable_ckpt_saving, optimizer_step, saved_step0_grad):
    """Save gradients for requested steps.

    Mapping:
    - step 0: gradients of the very first optimizer update (before optimizer.step()).
    - step N>0: gradients for optimizer update N (before optimizer.step()).
    """
    if disable_ckpt_saving:
        return saved_step0_grad

    if (0 in save_steps) and (optimizer_step == 0) and (not saved_step0_grad):
        save_kd_gradients(model, run_dir, 0)
        saved_step0_grad = True

    current_update_step = optimizer_step + 1
    if current_update_step in save_steps:
        save_kd_gradients(model, run_dir, current_update_step)

    return saved_step0_grad


def maybe_save_pretrain_checkpoint(model, run_dir, save_steps, disable_ckpt_saving):
    """Save a step=0 checkpoint before training when requested."""
    if (not disable_ckpt_saving) and (0 in save_steps):
        save_kd_checkpoint(model, run_dir, 0)


def evaluate_ce_val_loss(model, ce_val_loader, device):
    """Compute average CE loss on a validation loader."""
    was_training = model.training
    model.eval()
    total_val_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in ce_val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            total_val_loss += model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            ).loss.item()
            num_batches += 1
    if was_training:
        model.train()
    return total_val_loss / max(1, num_batches)


def is_math_teacher_dataset(kd_args: "KDTrainingConfig"):
    """Heuristic: treat teacher data as math when path contains 'math'."""
    return (
        kd_args.kd_loss_type != "ce"
        and bool(kd_args.teacher_data_dir)
        and "math" in kd_args.teacher_data_dir.lower()
    )


def _get_math_ground_truth(entry):
    for key in ("solution", "ground_truth", "answer"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _get_eval_prompt_ids(entry, tokenizer, max_length):
    prompt_ids = entry.get("prompt_ids")
    if isinstance(prompt_ids, list) and prompt_ids:
        return prompt_ids[:max_length]

    prompt_text = entry.get("prompt")
    if isinstance(prompt_text, str) and prompt_text.strip():
        ids = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_length,
            return_attention_mask=False,
        )["input_ids"]
        if ids:
            return ids
    return None


def _infer_eval_max_new_tokens(entry, prompt_ids, max_length):
    budget = max(1, max_length - len(prompt_ids))
    token_ids = entry.get("token_ids")
    if isinstance(token_ids, list) and token_ids:
        if len(token_ids) > len(prompt_ids):
            target_len = len(token_ids) - len(prompt_ids)
        else:
            target_len = len(token_ids)
        return max(1, min(target_len, budget))
    return min(256, budget)


def evaluate_math_val_accuracy(model, tokenizer, val_completions, device, max_length, accuracy_fn=None):
    """Generate answers on validation prompts and compute math verification accuracy."""
    if accuracy_fn is None:
        from open_r1.rewards import accuracy_reward as accuracy_fn

    was_training = model.training
    model.eval()

    scored = 0
    correct = 0.0

    with torch.no_grad():
        for entry in val_completions:
            ground_truth = _get_math_ground_truth(entry)
            prompt_ids = _get_eval_prompt_ids(entry, tokenizer, max_length)
            if ground_truth is None or prompt_ids is None:
                continue

            max_new_tokens = _infer_eval_max_new_tokens(entry, prompt_ids, max_length)

            input_ids = torch.tensor([prompt_ids], device=device)
            attention_mask = torch.ones_like(input_ids, device=device)
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

            generated_completion_ids = generated[0, input_ids.shape[1] :]
            generated_text = tokenizer.decode(generated_completion_ids, skip_special_tokens=True)

            reward = accuracy_fn(
                completions=[[{"content": generated_text}]],
                solution=[ground_truth],
            )[0]
            if reward is None:
                continue

            scored += 1
            correct += float(reward)

    if was_training:
        model.train()

    total = len(val_completions)
    skipped = total - scored
    accuracy = (correct / scored) if scored > 0 else None
    return {
        "accuracy": accuracy,
        "scored_samples": scored,
        "skipped_samples": skipped,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Teacher data loading
# ══════════════════════════════════════════════════════════════════════════════

def load_teacher_config(teacher_data_dir):
    config_path = os.path.join(teacher_data_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Teacher config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_completions(teacher_data_dir):
    path = os.path.join(teacher_data_dir, "completions.jsonl")
    completions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            completions.append(json.loads(line))
    return completions


def load_ce_sequences(kd_args: "KDTrainingConfig", tokenizer, total_required_samples: Optional[int] = None):
    """Load fixed-length CE token sequences from C4 or JSONL.

    Sequences are chunked to exactly `kd_args.ce_seq_len`.
    When source data is exhausted early, already-built chunks are cycled to
    satisfy the requested sample count.
    """
    from datasets import load_dataset as hf_load_dataset

    target_samples = total_required_samples if total_required_samples is not None else kd_args.ce_num_seqs
    if target_samples <= 0:
        raise ValueError(f"total_required_samples must be > 0, got {target_samples}")

    if kd_args.ce_data_source == "c4":
        dataset = hf_load_dataset("allenai/c4", "en", split="train", streaming=True)
        text_iter = (row["text"] for row in dataset)
    else:  # jsonl
        dataset = hf_load_dataset("json", data_files=kd_args.ce_data_path, split="train")
        text_iter = (row["text"] for row in dataset)

    sequences = []
    token_buffer = []
    cursor = 0
    eos_id = tokenizer.eos_token_id

    for text in text_iter:
        ids = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        if not ids:
            continue

        token_buffer.extend(ids)
        if eos_id is not None:
            token_buffer.append(eos_id)

        while cursor + kd_args.ce_seq_len <= len(token_buffer) and len(sequences) < target_samples:
            sequences.append(token_buffer[cursor : cursor + kd_args.ce_seq_len])
            cursor += kd_args.ce_seq_len

        # Avoid unbounded growth as we move the cursor forward.
        if cursor > 0 and cursor >= kd_args.ce_seq_len * 16:
            token_buffer = token_buffer[cursor:]
            cursor = 0

        if len(sequences) >= target_samples:
            break

    if not sequences:
        raise RuntimeError("load_ce_sequences: no sequences loaded — check ce_data_source/ce_data_path")

    if len(sequences) < target_samples:
        base = len(sequences)
        for i in range(target_samples - len(sequences)):
            sequences.append(sequences[i % base].copy())

    return sequences


def load_teacher_model(teacher_model_id, device):
    teacher = AutoModelForCausalLM.from_pretrained(teacher_model_id, torch_dtype=torch.bfloat16)
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher.to(device)


# ══════════════════════════════════════════════════════════════════════════════
#  Benchmarking (lm-eval-harness + C4/WikiText PPL)
# ══════════════════════════════════════════════════════════════════════════════

def _flatten_lm_eval_metrics(raw_results: dict) -> dict:
    """Reduce lm-eval-harness's per-task result dict to {task: primary_acc}.

    lm-eval returns a nested {task_name: {metric,stderr,...}} structure. We keep
    the main accuracy-style metric per task (acc_norm preferred, then acc), so
    the before/after comparison is a flat, readable mapping.
    """
    flat = {}
    for task, metrics in raw_results.items():
        if not isinstance(metrics, dict):
            continue
        value = None
        for key in ("acc_norm,none", "acc,none", "acc_norm", "acc", "exact_match,none", "exact_match"):
            if key in metrics and isinstance(metrics[key], (int, float)):
                value = float(metrics[key])
                break
        if value is None:
            # Fall back to the first numeric, non-stderr metric.
            for key, val in metrics.items():
                if "stderr" in key or not isinstance(val, (int, float)):
                    continue
                value = float(val)
                break
        if value is not None:
            flat[task] = value
    return flat


def run_benchmark(model, tokenizer, model_name, kd_args: "KDTrainingConfig", device):
    """Run C4/WikiText PPL + lm-eval-harness tasks; return a flat metrics dict.

    Keys are prefixed: ``ppl/<dataset>`` and ``lm_eval/<task>``.
    """
    metrics = {}

    # ── Perplexity (always includes C4) ──────────────────────────────────────
    ppl = evaluate_model_ppl(
        model,
        tokenizer,
        seqlen=kd_args.lm_eval_max_seqlen,
        seed=0,
        datasets=("wikitext2", "c4"),
        device=device,
    )
    for name, value in ppl.items():
        metrics[f"ppl/{name}"] = value
    print("[bench] PPL: " + ", ".join(f"{k}={v:.4f}" for k, v in ppl.items()))

    # ── lm-eval-harness tasks (hellaswag, mmlu, ...) ─────────────────────────
    tasks = [t.strip() for t in kd_args.lm_eval_tasks.split(",") if t.strip()]
    if tasks:
        from eval.lm_harness.eval import eval_tasks

        num_fewshot = 5 if any(t == "mmlu" for t in tasks) else 0
        was_training = model.training
        model.eval()
        try:
            raw = eval_tasks(
                model,
                model_name,
                tokenizer,
                tasks,
                limit=kd_args.lm_eval_limit,
                max_seqlen=kd_args.lm_eval_max_seqlen,
                batch_size=kd_args.lm_eval_batch_size,
                num_fewshot=num_fewshot,
            )
        finally:
            if was_training:
                model.train()
        for task, value in _flatten_lm_eval_metrics(raw).items():
            metrics[f"lm_eval/{task}"] = value
        print(
            "[bench] lm-eval: "
            + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k.startswith("lm_eval/"))
        )

    return metrics


def _print_benchmark_comparison(before: Optional[dict], after: dict):
    """Pretty-print a before/after benchmark comparison table."""
    keys = sorted(set((before or {}).keys()) | set(after.keys()))
    if not keys:
        return
    print("\n" + "=" * 72)
    print("  BENCHMARK COMPARISON (before compression  ->  after compress+train)")
    print("=" * 72)
    header = f"  {'metric':<24}{'before':>14}{'after':>14}{'delta':>14}"
    print(header)
    print("  " + "-" * 66)
    for key in keys:
        b = (before or {}).get(key)
        a = after.get(key)
        b_str = f"{b:.4f}" if isinstance(b, (int, float)) else "-"
        a_str = f"{a:.4f}" if isinstance(a, (int, float)) else "-"
        if isinstance(b, (int, float)) and isinstance(a, (int, float)):
            d_str = f"{a - b:+.4f}"
        else:
            d_str = "-"
        print(f"  {key:<24}{b_str:>14}{a_str:>14}{d_str:>14}")
    print("=" * 72 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main(script_args: KDScriptArguments, decomp_args: KDDecompositionConfig, kd_args: KDTrainingConfig):
    torch.manual_seed(script_args.seed)

    timestamp = time.strftime("%m%d-%H%M%S")
    model_stem = pathlib.Path(script_args.model_name_or_path).stem
    run_name = script_args.wandb_run_name or (
        f"{kd_args.kd_loss_type}_{decomp_args.train_mode}_calib-{decomp_args.calib_source}-{decomp_args.compression_ratio}_{script_args.lr:.1e}"
    )
    if script_args.name_suffix:
        run_name = f"{run_name}{script_args.name_suffix}"
    run_dir = os.path.join(kd_args.base_dir, f"{run_name}-{timestamp}")
    disable_ckpt_saving = is_ckpt_saving_disabled(kd_args.save_steps)
    if disable_ckpt_saving:
        print("Checkpoint saving disabled (save_steps=none). Run dir will be created only if other outputs are saved.")

    use_wandb = (not script_args.no_wandb) and (wandb is not None)
    if (not script_args.no_wandb) and (wandb is None):
        print("W&B is not installed; continuing without W&B logging.")
    if use_wandb:
        wandb.init(
            project=script_args.wandb_project,
            name=run_name,
            config={
                "model": script_args.model_name_or_path,
                "train_mode": decomp_args.train_mode,
                "compression_ratio": decomp_args.compression_ratio,
                "calib_source": decomp_args.calib_source,
                "decomp_mode": decomp_args.decomp_mode,
                "train_position": decomp_args.train_position,
                "kd_loss_type": kd_args.kd_loss_type,
                "ce_seq_len": kd_args.ce_seq_len,
                "ce_steps": kd_args.ce_steps,
                "lr": script_args.lr,
                "batch_size": script_args.batch_size,
                "gradient_accumulation_steps": script_args.gradient_accumulation_steps,
                "num_epochs": script_args.num_epochs,
                "optimizer": script_args.optimizer,
                "lr_scheduler": script_args.lr_scheduler,
                "warmup_ratio": script_args.warmup_ratio,
                "seed": script_args.seed,
                "run_dir": run_dir,
            },
            settings=wandb.Settings(console="redirect"),
        )
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("val/*", step_metric="train/step")
        wandb.define_metric("ppl/*")

    print(f"Run directory: {run_dir}")

    # ── Load teacher data (skipped for ce mode) ───────────────────────────────
    teacher_config = None
    train_completions = val_completions = None
    if kd_args.kd_loss_type != "ce":
        teacher_config = load_teacher_config(kd_args.teacher_data_dir)
        if kd_args.kd_loss_type == "kl" and kd_args.top_k > teacher_config["top_k"]:
            raise ValueError(
                f"top_k ({kd_args.top_k}) exceeds teacher data top_k ({teacher_config['top_k']})"
            )
        completions = load_completions(kd_args.teacher_data_dir)
        val_size = max(1, len(completions) // 5)
        train_completions = completions[:-val_size]
        val_completions = completions[-val_size:]
        print(f"Loaded {len(completions)} completions: {len(train_completions)} train, {len(val_completions)} val")

    # ── Load model + tokenizer ────────────────────────────────────────────────
    print(f"Loading model: {script_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path, torch_dtype=torch.bfloat16
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Baseline benchmark (uncompressed model) ───────────────────────────────
    baseline_metrics = None
    if kd_args.run_lm_eval and kd_args.eval_before_compression:
        print("[bench] Benchmarking base (uncompressed) model ...")
        baseline_metrics = run_benchmark(
            model, tokenizer, script_args.model_name_or_path, kd_args, device
        )
        if use_wandb:
            wandb.log({f"baseline/{k}": v for k, v in baseline_metrics.items()})

    # ── Decompose ─────────────────────────────────────────────────────────────
    model = decompose_model(model, tokenizer, decomp_args, train_completions=train_completions)
    compression_ppl_results = None

    if decomp_args.train_mode != "full" and decomp_args.eval_ppl_after_compression:
        print(
            f"[PPL] Running post-decomposition PPL on wikitext2,c4 "
            f"(seqlen={decomp_args.eval_ppl_seqlen}, seed={decomp_args.eval_ppl_seed}) ..."
        )
        compression_ppl_results = evaluate_model_ppl(
            model,
            tokenizer,
            seqlen=decomp_args.eval_ppl_seqlen,
            seed=decomp_args.eval_ppl_seed,
            datasets=("wikitext2", "c4"),
            device=device,
        )
        print("[PPL] " + ", ".join(f"{name}={value:.4f}" for name, value in compression_ppl_results.items()))
        ppl_path = os.path.join(run_dir, "compression_ppl.json")
        os.makedirs(run_dir, exist_ok=True)
        with open(ppl_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "datasets": ["wikitext2", "c4"],
                    "seqlen": decomp_args.eval_ppl_seqlen,
                    "seed": decomp_args.eval_ppl_seed,
                    "ppl": compression_ppl_results,
                },
                f,
                indent=2,
            )
        print(f"[PPL] Saved post-decomposition PPL results to {ppl_path}")

    # ── Optional online teacher ───────────────────────────────────────────────
    teacher_model = None
    if kd_args.kd_loss_type == "kl_online":
        print(f"Loading teacher model: {kd_args.teacher_model_id}")
        teacher_model = load_teacher_model(kd_args.teacher_model_id, device)

    # ── Build datasets + dataloaders ──────────────────────────────────────────
    if kd_args.kd_loss_type == "sft":
        train_dataset = KDSftDataset(train_completions, kd_args.max_length)
        val_dataset = KDSftDataset(val_completions, kd_args.max_length)
        collate_fn = build_kd_sft_collate_fn(tokenizer.pad_token_id)
    elif kd_args.kd_loss_type == "kl":
        train_dataset = KDKlDataset(train_completions, kd_args.teacher_data_dir, kd_args.top_k, kd_args.max_length)
        val_dataset = KDKlDataset(
            val_completions, kd_args.teacher_data_dir, kd_args.top_k, kd_args.max_length,
            index_offset=len(train_completions),
        )
        collate_fn = build_kd_kl_collate_fn(tokenizer.pad_token_id, kd_args.top_k)
    elif kd_args.kd_loss_type == "kl_online":
        train_dataset = KDOnlineDataset(train_completions, kd_args.max_length)
        val_dataset = KDOnlineDataset(val_completions, kd_args.max_length)
        collate_fn = build_kd_online_collate_fn(tokenizer.pad_token_id)
    else:  # ce
        print(f"[ce] Loading text sequences from {kd_args.ce_data_source} ...")
        ce_train_samples = (
            kd_args.ce_steps * script_args.gradient_accumulation_steps * script_args.batch_size
        )
        ce_val_samples = max(1, ce_train_samples // 5)
        ce_total_samples = ce_train_samples + ce_val_samples
        ce_sequences = load_ce_sequences(
            kd_args,
            tokenizer,
            total_required_samples=ce_total_samples,
        )
        train_dataset = KDCeDataset(ce_sequences[:ce_train_samples], kd_args.ce_seq_len)
        val_dataset = KDCeDataset(ce_sequences[ce_train_samples:], kd_args.ce_seq_len)
        collate_fn = build_kd_sft_collate_fn(tokenizer.pad_token_id)
        print(
            f"[ce] {len(ce_sequences)} sequences (seq_len={kd_args.ce_seq_len}): "
            f"{len(train_dataset)} train, {len(val_dataset)} val, steps={kd_args.ce_steps}"
        )

    train_loader = DataLoader(
        train_dataset, batch_size=script_args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=script_args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=collate_fn,
    )

    # Unified CE val loader for comparable val loss across kd_loss_type modes
    if kd_args.kd_loss_type in {"sft", "ce"}:
        ce_val_loader = val_loader
    else:
        ce_val_loader = DataLoader(
            KDSftDataset(val_completions, kd_args.max_length),
            batch_size=script_args.batch_size, shuffle=False,
            num_workers=0, pin_memory=True,
            collate_fn=build_kd_sft_collate_fn(tokenizer.pad_token_id),
        )

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    if kd_args.kd_loss_type == "ce":
        num_training_steps = kd_args.ce_steps
    else:
        num_training_steps = compute_num_training_steps(
            num_batches=len(train_loader),
            num_epochs=script_args.num_epochs,
            gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        )
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found after decomposition/trainability setup.")
    print(
        f"Trainable parameter tensors: {len(trainable_params)} "
        f"({sum(p.numel() for p in trainable_params):,} parameters)"
    )
    optimizer = build_optimizer(script_args, trainable_params)
    scheduler = build_lr_scheduler(script_args, optimizer, num_training_steps)
    save_steps = parse_save_steps(kd_args.save_steps, num_training_steps)

    # ── W&B ──────────────────────────────────────────────────────────────────
    shared_vocab_size = teacher_config.get("shared_vocab_size") if teacher_config else None
    if use_wandb and (shared_vocab_size is not None):
        wandb.config.update({"shared_vocab_size": shared_vocab_size}, allow_val_change=True)
    if use_wandb:
        if compression_ppl_results is not None:
            wandb.log({f"ppl/{k}": v for k, v in compression_ppl_results.items()})

    # Optional step-0 checkpoint (after decomposition, before training)
    maybe_save_pretrain_checkpoint(model, run_dir, save_steps, disable_ckpt_saving)

    # ── Initial val loss (before any optimizer step) ─────────────────────────
    initial_val_loss = evaluate_ce_val_loss(model, ce_val_loader, device)
    print(f"Initial val CE loss: {initial_val_loss:.4f}")
    if use_wandb:
        wandb.log({"val/ce_loss": initial_val_loss, "train/step": 0})

    # ── Training loop ─────────────────────────────────────────────────────────
    grad_accum = script_args.gradient_accumulation_steps
    optimizer_step = 0
    saved_step0_grad = False
    model.train()
    loop_epochs = script_args.num_epochs if kd_args.kd_loss_type != "ce" else 1

    for epoch in range(loop_epochs):
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
            batch = {k: v.to(device) for k, v in batch.items()}

            if kd_args.kd_loss_type in {"sft", "ce"}:
                loss = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                ).loss

            elif kd_args.kd_loss_type == "kl":
                student_logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                ).logits
                loss = compute_kl_loss(
                    student_logits,
                    batch["teacher_topk_values"],
                    batch["teacher_topk_indices"],
                    batch["response_mask"],
                )

            else:  # kl_online
                with torch.no_grad():
                    teacher_logits = teacher_model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    ).logits
                student_logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                ).logits
                loss = compute_online_kl_loss(
                    student_logits, teacher_logits, batch["response_mask"], shared_vocab_size
                )

            (loss / grad_accum).backward()

            if (batch_idx + 1) % grad_accum == 0:
                saved_step0_grad = maybe_save_gradient_snapshots(
                    model=model,
                    run_dir=run_dir,
                    save_steps=save_steps,
                    disable_ckpt_saving=disable_ckpt_saving,
                    optimizer_step=optimizer_step,
                    saved_step0_grad=saved_step0_grad,
                )
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                optimizer_step += 1

                if use_wandb:
                    wandb.log({"train/loss": loss.item(), "train/step": optimizer_step})

                if (not disable_ckpt_saving) and optimizer_step in save_steps:
                    save_kd_checkpoint(model, run_dir, optimizer_step)

                if kd_args.kd_loss_type == "ce" and optimizer_step >= num_training_steps:
                    break

        if kd_args.kd_loss_type == "ce" and optimizer_step >= num_training_steps:
            break

    # Final checkpoint (if not already saved)
    if (not disable_ckpt_saving) and (num_training_steps not in save_steps):
        save_kd_checkpoint(model, run_dir, "final")

    # ── Val CE loss ───────────────────────────────────────────────────────────
    val_loss = evaluate_ce_val_loss(model, ce_val_loader, device)
    print(f"Final val CE loss: {val_loss:.4f}")
    if use_wandb:
        wandb.log({"val/ce_loss": val_loss, "train/step": optimizer_step})

    # ── Math validation accuracy (end of training) ───────────────────────────
    # if is_math_teacher_dataset(kd_args):
    #     print("Running end-of-training math validation accuracy evaluation ...")
    #     math_metrics = evaluate_math_val_accuracy(
    #         model,
    #         tokenizer,
    #         val_completions,
    #         device=device,
    #         max_length=kd_args.max_length,
    #     )
    #     if math_metrics["accuracy"] is None:
    #         print(
    #             "[math-eval] No valid samples were scorable by math verifier "
    #             f"(skipped={math_metrics['skipped_samples']})."
    #         )
    #     else:
    #         print(
    #             f"[math-eval] val accuracy={math_metrics['accuracy']:.4f} "
    #             f"(scored={math_metrics['scored_samples']}, skipped={math_metrics['skipped_samples']})"
    #         )

    #     if use_wandb:
    #         math_payload = {
    #             "val/math_scored_samples": math_metrics["scored_samples"],
    #             "val/math_skipped_samples": math_metrics["skipped_samples"],
    #         }
    #         if math_metrics["accuracy"] is not None:
    #             math_payload["val/math_accuracy"] = math_metrics["accuracy"]
    #         wandb.log(math_payload)

    # ── Final benchmark (after compress + train) + comparison ─────────────────
    if kd_args.run_lm_eval:
        print("[bench] Benchmarking compressed + fine-tuned model ...")
        final_metrics = run_benchmark(
            model, tokenizer, script_args.model_name_or_path, kd_args, device
        )
        if use_wandb:
            wandb.log({f"final/{k}": v for k, v in final_metrics.items()})

        _print_benchmark_comparison(baseline_metrics, final_metrics)

        os.makedirs(run_dir, exist_ok=True)
        bench_path = os.path.join(run_dir, "benchmark_comparison.json")
        with open(bench_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": script_args.model_name_or_path,
                    "train_mode": decomp_args.train_mode,
                    "compression_ratio": decomp_args.compression_ratio,
                    "kd_loss_type": kd_args.kd_loss_type,
                    "tasks": kd_args.lm_eval_tasks,
                    "before_compression": baseline_metrics,
                    "after_compress_train": final_metrics,
                },
                f,
                indent=2,
            )
        print(f"[bench] Saved benchmark comparison to {bench_path}")

    if use_wandb:
        wandb.finish()

    if os.path.isdir(run_dir):
        print(f"Training complete. Outputs in {run_dir}")
    else:
        print("Training complete. No output directory was created.")


def parse_args_and_config(dataclass_types):
    """Parse a YAML ``--config`` file with CLI overrides into dataclasses.

    Drop-in replacement for trl's ``TrlParser.parse_args_and_config`` built on
    ``transformers.HfArgumentParser`` (an ``argparse.ArgumentParser`` subclass):
    values from the YAML file become argparse defaults, and any explicit CLI
    flag (``--key value``) overrides its YAML counterpart.
    """
    import argparse

    from transformers import HfArgumentParser

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    known, remaining = pre.parse_known_args()

    parser = HfArgumentParser(dataclass_types)

    if known.config:
        import yaml

        with open(known.config, "r", encoding="utf-8") as f:
            yaml_defaults = yaml.safe_load(f) or {}
        if not isinstance(yaml_defaults, dict):
            raise ValueError(f"Config {known.config!r} must contain a top-level mapping.")
        # YAML values become argparse defaults; CLI flags below override them.
        parser.set_defaults(**yaml_defaults)
        # Fields supplied by the YAML are no longer required on the CLI.
        for action in parser._actions:
            if action.dest in yaml_defaults:
                action.required = False

    return parser.parse_args_into_dataclasses(args=remaining)


if __name__ == "__main__":
    script_args, decomp_args, kd_args = parse_args_and_config(
        (KDScriptArguments, KDDecompositionConfig, KDTrainingConfig)
    )
    main(script_args, decomp_args, kd_args)