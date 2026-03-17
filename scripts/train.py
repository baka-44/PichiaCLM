"""
train.py
========
Training runner for Pichia-CLM Arch1.

Usage (local CPU – smoke-test only):
    python scripts/train.py --epochs 2 --batch_size 32 --checkpoint_dir checkpoints/

Usage (Colab GPU – full training):
    python scripts/train.py --checkpoint_dir /content/drive/MyDrive/PichiaCLM/checkpoints/

The script:
  1. Prepares data with an 80/20 protein-level train/test split
  2. Builds the dual-decoder Arch1 model
  3. Trains with early stopping + per-epoch checkpointing to checkpoint_dir
  4. Evaluates on the held-out test set after training
  5. Saves the final model + training history
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

# Local modules
import sys
sys.path.insert(0, os.path.dirname(__file__))
from data_prep import prepare_data, DATA_DIR
from model import (
    build_training_model, DEFAULT_HP,
    MAX_LENGTH, BATCH_SIZE, EPOCHS,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train Pichia-CLM Arch1")
    p.add_argument("--data_dir",       default=DATA_DIR,
                   help="Directory containing the 5 training CSV files")
    p.add_argument("--checkpoint_dir", default="checkpoints",
                   help="Directory to save model weights each epoch")
    p.add_argument("--epochs",     type=int, default=EPOCHS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--patience",   type=int, default=5,
                   help="Early stopping patience (epochs without val_loss improvement)")
    p.add_argument("--val_split",  type=float, default=0.20,
                   help="Fraction of training data used for validation during training")
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Input / target slicing  (replicates original paper's teacher-forcing setup)
# ---------------------------------------------------------------------------

def _strip_sequence(seq, end_token, inclusive):
    """Strip a padded 1-D integer array at its END token.

    inclusive=True  → keep the END token   (use for targets)
    inclusive=False → stop before END token (use for decoder inputs)
    Falls back to stripping trailing zeros when no END token is found.
    """
    end_positions = np.where(seq == end_token)[0]
    if len(end_positions) > 0:
        pos = int(end_positions[0])
        return seq[:pos + 1] if inclusive else seq[:pos]
    nz = np.where(seq != 0)[0]
    return seq[:int(nz[-1]) + 1] if len(nz) > 0 else seq[:1]


def make_dataset(AA, Cds, batch_size, shuffle=False, seed=42):
    """Build a tf.data.Dataset with per-batch dynamic padding.

    Each batch is padded only to the longest sequence it contains,
    rather than the global MAX_LENGTH=1000.  Average K. phaffii protein
    is ~350 AAs, so typical batches are ~400 tokens wide instead of 1000,
    giving roughly a 2–3x speedup when training with the standard GRU.

    Slicing / teacher-forcing logic mirrors make_inputs_targets exactly.
    """
    n = len(AA)

    enc_aa_pad    = AA[:,  1 : MAX_LENGTH + 1]   # [aa1..aaN, END=23]   strip incl END
    dec_codon_pad = Cds[:, 0 : MAX_LENGTH - 1]   # [START, c1..cN]      strip excl END
    dec_aa_pad    = AA[:,  0 : MAX_LENGTH]        # [START, aa1..aaN]    strip excl END
    tgt_codon_pad = Cds[:, 1 : MAX_LENGTH]        # [c1..cN, END=66]     strip incl END
    tgt_aa_pad    = AA[:,  1 : MAX_LENGTH + 1]    # [aa1..aaN, END=23]   strip incl END

    enc_aa_v    = [_strip_sequence(enc_aa_pad[i],    23, inclusive=True)  for i in range(n)]
    dec_codon_v = [_strip_sequence(dec_codon_pad[i], 66, inclusive=False) for i in range(n)]
    dec_aa_v    = [_strip_sequence(dec_aa_pad[i],    23, inclusive=False) for i in range(n)]
    tgt_codon_v = [_strip_sequence(tgt_codon_pad[i], 66, inclusive=True)  for i in range(n)]
    tgt_aa_v    = [_strip_sequence(tgt_aa_pad[i],    23, inclusive=True)  for i in range(n)]

    def gen():
        for e, dc, da, tc, ta in zip(enc_aa_v, dec_codon_v, dec_aa_v,
                                     tgt_codon_v, tgt_aa_v):
            yield ((e.astype(np.int32), dc.astype(np.int32), da.astype(np.int32)),
                   (tc.astype(np.int32), ta.astype(np.int32)))

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            (tf.TensorSpec(shape=(None,), dtype=tf.int32),
             tf.TensorSpec(shape=(None,), dtype=tf.int32),
             tf.TensorSpec(shape=(None,), dtype=tf.int32)),
            (tf.TensorSpec(shape=(None,), dtype=tf.int32),
             tf.TensorSpec(shape=(None,), dtype=tf.int32)),
        ),
    )

    if shuffle:
        ds = ds.shuffle(buffer_size=n, seed=seed, reshuffle_each_iteration=True)

    ds = ds.padded_batch(
        batch_size,
        padding_values=((np.int32(0), np.int32(0), np.int32(0)),
                        (np.int32(0), np.int32(0))),
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def make_inputs_targets(AA_tr, Cds_tr, max_length=MAX_LENGTH):
    """
    Slice padded arrays into the three model inputs and two targets.

    Encoder input  : AA tokens at positions 1:max_length+1
                     (skips column 0 which is always PAD in the padded array)
    Decoder codon  : CDS tokens at positions 0:max_length-1  (teacher forcing)
    Decoder AA aux : AA tokens at positions 0:max_length     (includes START token)

    Codon target   : CDS tokens at positions 1:max_length    (shifted +1 vs decoder input)
    AA target      : AA tokens  at positions 1:max_length+1  (shifted +1 vs decoder AA input)
    """
    enc_aa   = AA_tr[:,  1 : max_length + 1]   # (N, 1000)
    dec_codon = Cds_tr[:, 0 : max_length - 1]  # (N, 999)
    dec_aa   = AA_tr[:,  0 : max_length]        # (N, 1000)

    tgt_codon = Cds_tr[:, 1 : max_length]       # (N, 999)  ← codon labels
    tgt_aa    = AA_tr[:,  1 : max_length + 1]   # (N, 1000) ← AA labels

    return [enc_aa, dec_codon, dec_aa], [tgt_codon, tgt_aa]


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args):
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # --- Data -----------------------------------------------------------
    print("=" * 60)
    print("Step 1: Preparing data")
    print("=" * 60)
    data = prepare_data(data_dir=args.data_dir, verbose=True)

    train_inputs, train_targets = make_inputs_targets(data["AA_tr"], data["Cds_tr"])
    test_inputs,  test_targets  = make_inputs_targets(data["AA_ts"], data["Cds_ts"])

    print(f"\nTraining samples (proteins + augmented): {train_inputs[0].shape[0]:,}")
    print(f"Test samples (proteins only):            {test_inputs[0].shape[0]:,}")

    # --- Model ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2: Building model")
    print("=" * 60)
    model = build_training_model(DEFAULT_HP)
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")

    # --- Callbacks ------------------------------------------------------
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, "pichia_clm_arch1_ep{epoch:03d}.weights.h5")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(args.checkpoint_dir, "training_history.csv"),
            append=True,
        ),
    ]

    # --- Train ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: Training")
    print("=" * 60)
    history = model.fit(
        train_inputs,
        train_targets,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=args.val_split,
        callbacks=callbacks,
        verbose=1,
    )

    # --- Evaluate on held-out test set ----------------------------------
    print("\n" + "=" * 60)
    print("Step 4: Evaluating on held-out test proteins")
    print("=" * 60)
    results = model.evaluate(test_inputs, test_targets,
                             batch_size=args.batch_size, verbose=1)
    metric_names = model.metrics_names
    print("\nTest results:")
    for name, val in zip(metric_names, results):
        print(f"  {name}: {val:.4f}")

    # Codon accuracy is the first accuracy metric (output_codon_accuracy)
    codon_acc_idx = next(
        (i for i, n in enumerate(metric_names) if 'output_codon' in n and 'accuracy' in n),
        None
    )
    if codon_acc_idx is not None:
        print(f"\n>>> Codon prediction accuracy on test set: {results[codon_acc_idx]:.4f}")

    # --- Save final model -----------------------------------------------
    final_model_path = os.path.join(args.checkpoint_dir, "pichia_clm_arch1_final.weights.h5")
    model.save_weights(final_model_path)
    print(f"\nFinal weights saved to: {final_model_path}")

    # Save hyperparameters alongside weights for reproducibility
    hp_path = os.path.join(args.checkpoint_dir, "hyperparameters.json")
    with open(hp_path, "w") as f:
        json.dump({**DEFAULT_HP, "max_length": MAX_LENGTH,
                   "batch_size": args.batch_size, "epochs_run": len(history.history["loss"])},
                  f, indent=2)
    print(f"Hyperparameters saved to: {hp_path}")

    return model, history, data


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    model, history, data = train(args)
