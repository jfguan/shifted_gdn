"""Unified training script.

Usage:
    uv run train/run.py hebbian_18M train_stack_18M
    uv run train/run.py delta_hebbian_18M train_stack_18M
    uv run train/run.py gdn_18M train_stack_18M
"""

import argparse
import json
import math
import os
import time
from dataclasses import replace

import torch

from data import load_dataset, DataLoader
from models import build_model
import train.configs as C

MODELS = {
    "gdn_18M": C.GDN_18M,
    "shifted_gdn_18M": C.SHIFTED_GDN_18M,
    "gdn_100M": C.GDN_100M,
    "shifted_gdn_100M": C.SHIFTED_GDN_100M,
    "transformer_18M": C.TRANSFORMER_18M,
    "transformer_ts_18M": C.TRANSFORMER_TS_18M,
}

TRAINS = {
    "train_stack_18M": C.TRAIN_STACK_18M,
    "train_stack_100M": C.TRAIN_STACK_100M,
}


def main():
    torch.manual_seed(42)

    # setup
    model_config, train_config, resume = parse_args()
    device = setup_device()
    dataset, train_loader, val_loader = setup_data(train_config)
    model_config.vocab_size = dataset.vocab_size
    model = build_model(model_config).to(device)
    if device == "cuda":
        model = torch.compile(model)
    optimizer = configure_optimizers(
        model, train_config.lr, use_fused=(device == "cuda")
    )
    checkpoint_path = f"checkpoints/{model_config.name}.pt"

    def save(step, path):
        torch.save(
            {
                "model": unwrap(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_config": model_config,
                "step": step,
            },
            path,
        )
        print(f"  -> {path}")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("histories", exist_ok=True)
    log_path = f"histories/{model_config.name}.jsonl"
    if not resume:
        open(log_path, "w").close()

    def log(step, train_loss, step_ms, val_loss=None):
        tokens = step * tokens_per_step
        val_str = f" | val {val_loss:.4f}" if val_loss is not None else ""
        print(
            f"step {step} | loss {train_loss:.4f}{val_str} | {tokens / 1e6:.1f}M tok | {step_ms:.0f}ms"
        )

        entry = {
            "step": step,
            "train_loss": round(train_loss, 4),
            "tokens": tokens,
            "step_ms": round(step_ms, 2),
        }
        if val_loss is not None:
            entry["val_loss"] = round(val_loss, 4)
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # print stats
    parameter_sum = sum(p.numel() for p in model.parameters())
    print(f"{model_config.name} | {parameter_sum / 1e6:.1f}M params | {device}")
    start_step = resume_from(model, optimizer, checkpoint_path, device) if resume else 1
    print(
        f"Steps {start_step} -> {train_config.steps} | B={train_config.batch_size}x{train_config.grad_accum} T={train_config.seq_len} lr={train_config.lr}"
    )

    # training loop
    max_lr = train_config.lr
    min_lr = max_lr * 0.1
    warmup_steps = train_config.warmup
    decay_steps = max(train_config.steps - warmup_steps, 1)
    tokens_per_step = (
        train_config.batch_size * train_config.seq_len * train_config.grad_accum
    )

    end_step = train_config.steps
    if train_config.max_steps_per_run is not None:
        end_step = min(end_step, start_step + train_config.max_steps_per_run)

    for step in range(start_step, end_step + 1):
        t0 = time.time()

        # lr schedule: linear warmup then cosine decay
        if step <= warmup_steps:
            lr = max_lr * step / warmup_steps
        else:
            progress = (step - warmup_steps) / decay_steps
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            lr = min_lr + (max_lr - min_lr) * cosine_decay
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # forward + backward with grad accumulation
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(train_config.grad_accum):
            x, y = train_loader.batch()
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss = loss / train_config.grad_accum
            loss.backward()
            loss_accum += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if device == "mps" and step % 1000 == 0:
            torch.mps.empty_cache()

        last_step = step == end_step

        # eval
        eval_step = step % train_config.eval_interval == 0 or last_step
        val_loss = evaluate(model, val_loader, device) if eval_step else None

        # log
        step_ms = (time.time() - t0) * 1000
        log(step, loss_accum, step_ms=step_ms, val_loss=val_loss)

        # save
        if step % train_config.ckpt_interval == 0 or last_step:
            save(step, checkpoint_path)

    # sample
    raw_model = unwrap(model)
    prompt = """def fizzbuzz(n):
    for i in range(1, n + 1):
        if i % 3 == 0:
            print("fizz")
        elif i % 5 == 0:
            print("buzz")
        elif i % 15 == 0:
            print("""
    try:
        print(
            f"Sample:\n{sample(raw_model, dataset.encode, dataset.decode, device, prompt=prompt, n=300)}"
        )
    except NotImplementedError:
        print("(sampling not supported for this model)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS.keys())
    parser.add_argument("train", choices=TRAINS.keys())
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()
    model_config = replace(MODELS[args.model])
    train_config = replace(TRAINS[args.train])
    # run name: <model>_<dataset>_<size>, e.g. "hebbian_pg19_18M"
    dataset = train_config.dataset.value  # "pg19" or "the_stack"
    model_config.name = f"{args.model}_{dataset}"
    return model_config, train_config, args.resume


def setup_device():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def setup_data(train_config):
    dataset = load_dataset(train_config.dataset)

    train_loader = DataLoader(
        dataset.train, train_config.batch_size, train_config.seq_len
    )
    val_loader = DataLoader(dataset.val, train_config.batch_size, train_config.seq_len)

    return dataset, train_loader, val_loader


def configure_optimizers(model, lr, use_fused):
    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": 0.1},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), fused=use_fused)


def resume_from(model, optimizer, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Load in model state
    unwrap(model).load_state_dict(checkpoint["model"])

    # Load in optimizer
    optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint["step"]


def unwrap(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


@torch.no_grad()
def evaluate(model, loader, device, steps=20):
    model.eval()

    total_loss = 0.0
    for _ in range(steps):
        x, y = loader.batch()

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            _, loss = model(x.to(device), y.to(device))

        total_loss += loss.item()
    model.train()

    return total_loss / steps


@torch.no_grad()
def sample(model, encode, decode, device, prompt="", n=200, temperature=0.8):
    model.eval()
    prompt_ids = encode(prompt) if prompt else [0]
    token = torch.tensor([prompt_ids[0]], dtype=torch.long, device=device)
    states = None
    generated = []

    for i in range(len(prompt_ids) - 1 + n):
        logits, states = model.step(token, states=states)
        if i < len(prompt_ids) - 1:
            # still feeding prompt
            token = torch.tensor([prompt_ids[i + 1]], dtype=torch.long, device=device)
        else:
            # sampling
            token = torch.multinomial(
                torch.softmax(logits / temperature, dim=-1), 1
            ).squeeze(-1)
            generated.append(token.item())

    model.train()
    return prompt + decode(generated)


if __name__ == "__main__":
    main()
