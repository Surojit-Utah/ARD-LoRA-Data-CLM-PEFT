import os
import subprocess
from datasets import load_dataset


def _maybe_clone_repo(repo_url: str, dest: str = '/content/bayesian_peft'):
    # If repo_url looks like a GitHub URL, clone into dest if dest does not already exist
    if repo_url.startswith('http') and 'github.com' in repo_url:
        if not os.path.exists(dest):
            subprocess.run(["git", "clone", repo_url, dest], check=True)
        return dest
    # otherwise assume repo_url is a local path
    return repo_url


def load_bayesian_peft_dataset(repo_url: str, dataset_path: str, tokenizer=None, max_len=2048, sample_size: int = None, streaming: bool = False, dest: str = '/content/bayesian_peft', save_sample_path: str = None):
    """Load Bayesian-PEFT dataset files. If repo_url is a GitHub URL and running in Colab, this will clone the repo.

    Returns tokenized train and validation datasets ready for causal LM training.
    """
    base = _maybe_clone_repo(repo_url, dest)
    train_file = os.path.join(base, dataset_path, "train.jsonl")
    val_file = os.path.join(base, dataset_path, "validation.jsonl")

    data_files = {}
    if os.path.exists(train_file):
        data_files["train"] = train_file
    if os.path.exists(val_file):
        data_files["validation"] = val_file

    if not data_files:
        raise FileNotFoundError(f"No dataset files found at {train_file} or {val_file}")

    # streaming: return small in-memory Dataset built from iterator samples
    if streaming:
        ds_stream = load_dataset("json", data_files=data_files, streaming=True)
        def take_split(split_name):
            it = ds_stream[split_name]
            if sample_size is None:
                # return iterator (streaming)
                return it
            else:
                from itertools import islice
                items = list(islice(it, sample_size))
                from datasets import Dataset
                return Dataset.from_list(items)

        train_ds = take_split("train") if "train" in ds_stream else None
        val_ds = take_split("validation") if "validation" in ds_stream else None

        # If tokenizer provided and we materialized a small Dataset, tokenize and apply prompt-masking
        if tokenizer is not None and isinstance(train_ds, object) and not hasattr(train_ds, '__iter__'):
            # reuse tokenization logic below by mapping after transforming examples to prompt/full parts
            def build_prompt_parts_stream(ex):
                # build prompt_text and response_text separately so we can mask prompt tokens
                if "instruction" in ex and "output" in ex:
                    prompt = ex["instruction"]
                    if ex.get("input"):
                        prompt += "\n" + ex["input"]
                    response = ex["output"]
                    full = prompt + "\n" + response
                    return {"prompt_text": prompt, "full_text": full}
                if "text" in ex:
                    return {"prompt_text": "", "full_text": ex["text"]}
                return {"prompt_text": "", "full_text": str(ex)}

            train_ds = train_ds.map(build_prompt_parts_stream, remove_columns=train_ds.column_names)
            # apply tokenization + masking
            def tokenize_and_mask(batch):
                # batch contains 'prompt_text' and 'full_text'
                full_tok = tokenizer(batch["full_text"], truncation=True, max_length=max_len, padding="max_length")
                # tokenize prompts without padding to get true prompt lengths
                prompt_tok = tokenizer(batch["prompt_text"], truncation=True, max_length=max_len)
                labels = [ids.copy() for ids in full_tok["input_ids"]]
                for i, pids in enumerate(prompt_tok["input_ids"]):
                    p_len = len(pids)
                    fid = full_tok["input_ids"][i]
                    # mask prompt token positions
                    for j in range(min(p_len, len(fid))):
                        labels[i][j] = -100
                full_tok["labels"] = labels
                return full_tok

            train_ds = train_ds.map(tokenize_and_mask, batched=True, remove_columns=[c for c in train_ds.column_names])
    else:
        ds = load_dataset("json", data_files=data_files)

        def build_prompt(ex):
            # return both prompt_text and full_text so tokenization can mask prompts
            if "instruction" in ex and "output" in ex:
                prompt = ex["instruction"]
                if ex.get("input"):
                    prompt += "\n" + ex["input"]
                response = ex["output"]
                full = prompt + "\n" + response
                return {"prompt_text": prompt, "full_text": full}
            if "text" in ex:
                return {"prompt_text": "", "full_text": ex["text"]}
            return {"prompt_text": "", "full_text": str(ex)}

        # If sample_size is specified, subsample using select
        if sample_size is not None:
            train_ds = ds["train"].select(range(min(sample_size, len(ds["train"])))) if "train" in ds else None
            val_ds = ds["validation"].select(range(min(int(sample_size*0.05), len(ds.get("validation", []))))) if "validation" in ds else None
        else:
            train_ds = ds.get("train")
            val_ds = ds.get("validation")

        if train_ds is not None:
            train_ds = train_ds.map(build_prompt, remove_columns=train_ds.column_names)
        if val_ds is not None:
            val_ds = val_ds.map(build_prompt, remove_columns=val_ds.column_names)

        if tokenizer is not None and train_ds is not None:
            def tokenize_and_mask_batch(batch):
                # batch has 'prompt_text' and 'full_text'
                full_tok = tokenizer(batch["full_text"], truncation=True, max_length=max_len, padding="max_length")
                # get prompt lengths by tokenizing prompts (no padding)
                prompt_tok = tokenizer(batch["prompt_text"], truncation=True, max_length=max_len)
                labels = [ids.copy() for ids in full_tok["input_ids"]]
                for i, pids in enumerate(prompt_tok["input_ids"]):
                    p_len = len(pids)
                    fid = full_tok["input_ids"][i]
                    for j in range(min(p_len, len(fid))):
                        labels[i][j] = -100
                full_tok["labels"] = labels
                return full_tok

            train_ds = train_ds.map(tokenize_and_mask_batch, batched=True, remove_columns=["prompt_text", "full_text"]) if train_ds is not None else None
            val_ds = val_ds.map(tokenize_and_mask_batch, batched=True, remove_columns=["prompt_text", "full_text"]) if val_ds is not None else None

    # Optionally save a sampled subset to disk
    if save_sample_path is not None and train_ds is not None:
        import json
        os.makedirs(os.path.dirname(save_sample_path), exist_ok=True)
        with open(save_sample_path, 'w', encoding='utf-8') as f:
            for ex in train_ds[:min(1000, len(train_ds))]:
                f.write(json.dumps(ex) + '\n')

    return train_ds, val_ds
