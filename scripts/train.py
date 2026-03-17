"""
train.py
========
Training utilities for Pichia-CLM Arch1 — PyTorch.

Usage:
    python scripts/train.py --checkpoint_dir /content/drive/MyDrive/PichiaCLM/checkpoints
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(__file__))
from data_prep import prepare_data, DATA_DIR
from model import (
    build_training_model, count_parameters,
    DEFAULT_HP, MAX_LENGTH, BATCH_SIZE, EPOCHS, LEARNING_RATE,
)

PAD_IDX = 0


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def make_inputs_targets(AA_tr, Cds_tr, max_length=MAX_LENGTH):
    """
    Slice padded arrays into model inputs and prediction targets.

    Encoder input   : AA tokens positions 1:max_length+1
    Decoder codon   : CDS tokens positions 0:max_length-1  (teacher forcing)
    Decoder AA aux  : AA tokens positions 0:max_length     (includes START)
    Codon target    : CDS tokens positions 1:max_length    (shifted +1)
    AA target       : AA tokens positions 1:max_length+1   (shifted +1)
    """
    enc_aa    = AA_tr[:,  1 : max_length + 1]   # (N, 1000)
    dec_codon = Cds_tr[:, 0 : max_length - 1]   # (N, 999)
    dec_aa    = AA_tr[:,  0 : max_length]        # (N, 1000)
    tgt_codon = Cds_tr[:, 1 : max_length]        # (N, 999)
    tgt_aa    = AA_tr[:,  1 : max_length + 1]    # (N, 1000)
    return [enc_aa, dec_codon, dec_aa], [tgt_codon, tgt_aa]


def make_dataset(AA, Cds, batch_size=None, shuffle=False, seed=42):
    """Return (inputs, targets) as numpy arrays — same interface as Keras version."""
    if shuffle:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(AA))
        AA, Cds = AA[idx], Cds[idx]
    return make_inputs_targets(AA, Cds)


def make_dataloader(inputs, targets, batch_size, shuffle=False):
    """Wrap numpy arrays into a PyTorch DataLoader."""
    tensors = [torch.from_numpy(np.asarray(x)).long() for x in inputs + targets]
    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      pin_memory=True, num_workers=0)


# ---------------------------------------------------------------------------
# Loss / accuracy
# ---------------------------------------------------------------------------

def compute_loss(crit_codon, crit_aa, codon_logits, aa_logits, tgt_codon, tgt_aa):
    """Total loss = codon CE + AA CE (both averaged over non-PAD positions)."""
    B, T_c, V_c = codon_logits.shape
    B, T_a, V_a = aa_logits.shape
    loss_c = crit_codon(codon_logits.reshape(B * T_c, V_c), tgt_codon.reshape(B * T_c))
    loss_a = crit_aa(aa_logits.reshape(B * T_a, V_a),       tgt_aa.reshape(B * T_a))
    return loss_c + loss_a


def masked_accuracy(logits, targets):
    """Fraction of non-PAD positions correctly predicted."""
    pred    = logits.argmax(dim=-1)
    mask    = targets != PAD_IDX
    correct = ((pred == targets) & mask).sum().item()
    total   = mask.sum().item()
    return correct / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Epoch runner
# ---------------------------------------------------------------------------

def run_epoch(model, loader, device, crit_codon, crit_aa,
              optimizer=None, scaler=None):
    """
    Run one epoch.  Trains if optimizer is provided, evaluates otherwise.
    Returns (loss, codon_accuracy, aa_accuracy) — averages over batches.
    """
    training = optimizer is not None
    model.train(training)

    total_loss = total_c_acc = total_a_acc = 0.0
    n = 0

    with torch.set_grad_enabled(training):
        for enc_aa, dec_codon, dec_aa, tgt_codon, tgt_aa in loader:
            enc_aa    = enc_aa.to(device, non_blocking=True)
            dec_codon = dec_codon.to(device, non_blocking=True)
            dec_aa    = dec_aa.to(device, non_blocking=True)
            tgt_codon = tgt_codon.to(device, non_blocking=True)
            tgt_aa    = tgt_aa.to(device, non_blocking=True)

            use_amp = training and (device.type == 'cuda')

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                codon_logits, aa_logits = model(enc_aa, dec_codon, dec_aa)
                loss = compute_loss(crit_codon, crit_aa,
                                    codon_logits, aa_logits,
                                    tgt_codon, tgt_aa)

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss  += loss.item()
            total_c_acc += masked_accuracy(codon_logits.detach(), tgt_codon)
            total_a_acc += masked_accuracy(aa_logits.detach(),    tgt_aa)
            n += 1

    return total_loss / n, total_c_acc / n, total_a_acc / n


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')
    if device.type == 'cuda':
        print(f'GPU    : {torch.cuda.get_device_name(0)}')
        print(f'VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    # ── Data ────────────────────────────────────────────────────────────────
    print('\nPreparing data...')
    data = prepare_data(data_dir=args.data_dir, verbose=True)

    n_total = len(data['AA_tr'])
    n_train = int(n_total * (1 - args.val_split))

    tr_in,  tr_tgt  = make_dataset(
        data['AA_tr'][:n_train], data['Cds_tr'][:n_train], shuffle=True, seed=args.seed)
    val_in, val_tgt = make_inputs_targets(
        data['AA_tr'][n_train:], data['Cds_tr'][n_train:])
    ts_in,  ts_tgt  = make_inputs_targets(data['AA_ts'], data['Cds_ts'])

    train_loader = make_dataloader(tr_in,  tr_tgt,  args.batch_size, shuffle=True)
    val_loader   = make_dataloader(val_in, val_tgt, args.batch_size, shuffle=False)
    test_loader  = make_dataloader(ts_in,  ts_tgt,  args.batch_size, shuffle=False)

    print(f'Train {n_train:,}  |  Val {n_total - n_train:,}  |  Test {len(data["AA_ts"]):,}')

    # ── Model ───────────────────────────────────────────────────────────────
    model = build_training_model(DEFAULT_HP, device=device)
    print(f'Parameters: {count_parameters(model):,}')

    optimizer  = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler     = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    crit_codon = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    crit_aa    = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # ── Training loop ───────────────────────────────────────────────────────
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    csv_path  = os.path.join(args.checkpoint_dir, 'training_history.csv')
    best_path = os.path.join(args.checkpoint_dir, 'best_weights.pt')

    best_val_loss    = float('inf')
    patience_counter = 0

    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow([
            'epoch', 'loss', 'codon_acc', 'aa_acc',
            'val_loss', 'val_codon_acc', 'val_aa_acc', 'epoch_s'])

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss,  tr_c_acc,  tr_a_acc  = run_epoch(
            model, train_loader, device, crit_codon, crit_aa, optimizer, scaler)
        val_loss, val_c_acc, val_a_acc = run_epoch(
            model, val_loader, device, crit_codon, crit_aa)

        elapsed = time.time() - t0

        print(f'Epoch {epoch:3d}/{args.epochs}  '
              f'loss={tr_loss:.4f} codon_acc={tr_c_acc:.4f}  '
              f'val_loss={val_loss:.4f} val_codon_acc={val_c_acc:.4f}  '
              f'{elapsed:.0f}s')

        # Per-epoch checkpoint
        ckpt = os.path.join(args.checkpoint_dir, f'pichia_clm_ep{epoch:03d}.pt')
        torch.save(model.state_dict(), ckpt)

        # CSV log
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch, tr_loss, tr_c_acc, tr_a_acc,
                val_loss, val_c_acc, val_a_acc, elapsed])

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
            print(f'  ✓ Best model saved (val_loss={val_loss:.4f})')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'Early stopping at epoch {epoch}')
                break

    # ── Load best weights and evaluate on test set ──────────────────────────
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    ts_loss, ts_c_acc, ts_a_acc = run_epoch(
        model, test_loader, device, crit_codon, crit_aa)
    print(f'\nTest  loss={ts_loss:.4f}  codon_acc={ts_c_acc:.4f}  aa_acc={ts_a_acc:.4f}')

    # Save final model
    final_path = os.path.join(args.checkpoint_dir, 'pichia_clm_final.pt')
    torch.save(model.state_dict(), final_path)
    with open(os.path.join(args.checkpoint_dir, 'hyperparameters.json'), 'w') as f:
        json.dump({**DEFAULT_HP, 'max_length': MAX_LENGTH,
                   'batch_size': args.batch_size}, f, indent=2)
    print(f'Saved final model: {final_path}')

    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',       default=DATA_DIR)
    p.add_argument('--checkpoint_dir', default='checkpoints')
    p.add_argument('--epochs',     type=int,   default=EPOCHS)
    p.add_argument('--batch_size', type=int,   default=BATCH_SIZE)
    p.add_argument('--lr',         type=float, default=LEARNING_RATE)
    p.add_argument('--patience',   type=int,   default=5)
    p.add_argument('--val_split',  type=float, default=0.20)
    p.add_argument('--seed',       type=int,   default=42)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
