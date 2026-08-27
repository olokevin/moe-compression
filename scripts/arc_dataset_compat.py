"""Make ``allenai/ai2_arc`` loadable under ``datasets==3.6.0``.

The Hub card for ``allenai/ai2_arc`` now declares its features with the ``List`` type,
which was introduced in ``datasets`` 4.x. Loading it under the pinned 3.6.0 raises

    ValueError: Feature type 'List' not found. Available feature types: [... 'LargeList' ...]

The *data* is fine -- the repo holds plain per-split parquet files -- only the card's
embedded ``dataset_info`` is unreadable. So this shim redirects that one dataset id to
its parquet files and leaves everything else alone.

Deliberately a shim and not a local task YAML: lm-eval's ``arc_challenge`` config
(``doc_to_text``, ``doc_to_choice``, the ``acc``/``acc_norm`` metrics) stays exactly the
built-in one, so the numbers remain comparable to any other lm-eval run. Only the
dataset *transport* changes.

Alternative, if the pin ever moves: ``datasets>=4`` reads the card directly and this
shim becomes a no-op (it only fires when the direct load fails).
"""

import datasets

from src.base.shared_utils import _print

__all__ = ["install_arc_parquet_shim"]

_PARQUET = {
    "train": "{name}/train-00000-of-00001.parquet",
    "validation": "{name}/validation-00000-of-00001.parquet",
    "test": "{name}/test-00000-of-00001.parquet",
}
_REPO = "allenai/ai2_arc"


def install_arc_parquet_shim(verbose=True):
    """Idempotently patch ``datasets.load_dataset`` for ``allenai/ai2_arc``."""
    if getattr(datasets.load_dataset, "_arc_shim", False):
        return
    original = datasets.load_dataset

    def patched(path=None, name=None, *args, **kwargs):
        if path != _REPO:
            return original(path, name, *args, **kwargs)
        try:
            return original(path, name, *args, **kwargs)
        except ValueError as e:
            if "Feature type" not in str(e):
                raise
        from huggingface_hub import hf_hub_download

        files = {
            split: hf_hub_download(_REPO, tmpl.format(name=name), repo_type="dataset")
            for split, tmpl in _PARQUET.items()
        }
        # Only transport-level kwargs survive; a builder-specific one (data_dir,
        # download_mode as an enum) would not mean the same thing to the parquet builder.
        keep = {k: v for k, v in kwargs.items() if k in ("split", "cache_dir")}
        if verbose:
            _print(f"[arc-compat] {_REPO}/{name}: card unreadable under "
                   f"datasets {datasets.__version__}; loading its parquet files directly")
        return original("parquet", data_files=files, **keep)

    patched._arc_shim = True
    datasets.load_dataset = patched
