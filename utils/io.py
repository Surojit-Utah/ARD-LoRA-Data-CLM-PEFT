import os
import gc
import torch


def get_output_dirs(runId=1, base_dir=None):
    if base_dir is None:
        base_dir = os.path.join(os.getcwd(), "run_outputs")

    log_dir = os.path.join(base_dir, "run_" + str(runId))
    os.makedirs(log_dir, exist_ok=True)

    output_dir = os.path.join(log_dir, "latent_images")
    model_ckpnt_dir = os.path.join(log_dir, "checkpoint")
    tb_log_dir = os.path.join(log_dir, "tb_logs")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_ckpnt_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)

    return output_dir, model_ckpnt_dir, tb_log_dir


def free_memory():
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
