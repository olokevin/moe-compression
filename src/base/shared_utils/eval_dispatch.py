
def _get_lm_harness_tasks():
    return ["human_eval", "humaneval", "gsm8k", "gsm8k_cot", "gsm8k_cot_fs", "c4", "wikitext2", "wikitext", "commonsenseqa", "mmlu", "hellaswag", "piqa", "boolq", "winogrande", "arc_easy", "arc_challenge", "openbookqa"]


def _flatten_eval_results(results):
    """Flatten the nested lm-eval results dict into scalar wandb metrics.

    lm-eval returns e.g. {"mmlu": {"acc,none": 0.81, "acc_stderr,none": 0.003,
    "alias": "mmlu"}, "mmlu_humanities": {...}, ...}. We keep the numeric leaves
    and namespace them as ``eval/<task>/<metric>`` so wandb can chart them.
    """
    flat = {}
    for task, payload in results.items():
        if isinstance(payload, dict):
            for metric, value in payload.items():
                if isinstance(value, bool):
                    continue
                try:
                    flat[f"eval/{task}/{metric.replace(',', '_')}"] = float(value)
                except (TypeError, ValueError):
                    continue  # skip non-numeric leaves (e.g. "alias")
        else:
            try:
                flat[f"eval/{task}"] = float(payload)
            except (TypeError, ValueError):
                continue
    return flat


def _log_results_to_wandb(args, results):
    """Best-effort wandb logging for the eval path.

    Gated on ``args.use_wandb``; never raises (wandb missing / offline / auth
    failure just prints a warning and continues). Mirrors the wandb env setup
    that ``src/train/utils/training_config.py`` does for the training path so
    eval runs land in the same project.
    """
    if not getattr(args, "use_wandb", False):
        return
    try:
        import os
        import wandb
    except ImportError:
        print("[wandb] use_wandb=True but wandb is not installed; skipping eval logging.")
        return

    try:
        run = wandb.run
        if run is None:
            run = wandb.init(
                project=getattr(args, "wandb_project", "slimmoe_kd"),
                name=getattr(args, "wandb_name", None),
                config={
                    "model_name_or_path": getattr(args, "model_name_or_path", None),
                    "eval_task_names": getattr(args, "eval_task_names", None),
                    "num_fewshot": getattr(args, "num_fewshot", None),
                    "eval_sample_limit": getattr(args, "eval_sample_limit", None),
                    "prune_kwargs": getattr(args, "prune_kwargs", None),
                    "real_slim": getattr(args, "real_slim", None),
                    "resume_path": getattr(args, "resume_path", None),
                },
                job_type="eval",
                reinit=False,
            )
        flat = _flatten_eval_results(results)
        if flat:
            run.log(flat)
            run.summary.update(flat)
        print(f"[wandb] logged {len(flat)} eval metrics to {run.project}/{run.name}")
    except Exception as e:  # pragma: no cover - logging must never break eval
        print(f"[wandb] eval logging failed ({type(e).__name__}: {e}); continuing.")


def eval_dispatch(args, model, tokenizer, verbose=False):
    _tasks = [task for task in args.eval_task_names.split(",")]
    results = {}
    if any(task.lower() in _get_lm_harness_tasks() for task in _tasks):
        from eval.lm_harness.eval import eval_fn
        res = eval_fn(args, model, tokenizer, _tasks, verbose=verbose)
        results.update(res)

    _log_results_to_wandb(args, results)

    return results
