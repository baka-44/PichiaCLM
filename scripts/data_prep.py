"""
data_prep.py
============
Clean data preparation for Pichia-CLM replication.

Key fixes vs original DataPrep_AllData.py:
  - Proper 80/20 train/test split on PROTEIN sequences before augmentation,
    so the held-out test set contains real proteins (not just 67 single-codon pairs).
  - AA maxlen reduced from 10001 → 1002 (covers every protein in the dataset;
    the model always slices to Max_length=1000 anyway, saving ~10x encoder RAM).
  - Augmented single-codon pairs added to TRAINING set only.
  - All tokeniser dicts returned so inference code can reuse them.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__),
                        "../Model_PichiaCLM/Training/AllData")

CSV_FILES = [
    "CBS7435.csv",
    "GS115_ext.csv",
    "Kpastoris_WT.csv",
    "Kphaffi_GS115_int.csv",
    "Kphaffi_WT.csv",
]

# Vocabulary sizes (fixed by the codon table, not tunable)
AA_VOCAB_SIZE  = 25   # indices 0-24  (PAD=0, AAs=1-20, ambig=21, stop*=22, END=23, START=24)
DNA_VOCAB_SIZE = 67   # indices 0-66  (PAD=0, codons=1-64, START=65, END=66)

# Padding lengths
AA_MAXLEN  = 1002   # covers longest K.phaffii protein (4951 AA) with START/END headroom
                    # NOTE: model always slices to Max_length=1000 at training time
CDS_MAXLEN = 1000   # same as model Max_length; CDS > 998 codons gets truncated at decoder

# Augmentation repetitions (replicates original paper)
N_REPS = 25

# Reproducible split
RANDOM_STATE = 42
TEST_FRACTION = 0.20


# ---------------------------------------------------------------------------
# Tokenisation helpers (exact mappings from original paper)
# ---------------------------------------------------------------------------

def get_aa_dict() -> dict:
    return {
        'A': 1,  'C': 2,  'D': 3,  'E': 4,  'F': 5,
        'G': 6,  'H': 7,  'I': 8,  'K': 9,  'L': 10,
        'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
        'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,
        # ambiguous / non-standard residues → same token
        'X': 21, 'Z': 21, 'B': 21, 'U': 21, 'O': 21,
        '*': 22,  # stop codon when written as amino-acid character
        # special model tokens
        # END = 23, START = 24, PAD = 0  (added by tokenise_aa, not in dict)
    }


def get_aa_codon_lists() -> tuple[list, list]:
    """Return (AA_list, codon_list) with one entry per codon (same order as paper)."""
    dic_aa_codon = {
        'A': ['GCT', 'GCC', 'GCA', 'GCG'],
        'C': ['TGT', 'TGC'],
        'D': ['GAT', 'GAC'],
        'E': ['GAA', 'GAG'],
        'F': ['TTT', 'TTC'],
        'G': ['GGT', 'GGA', 'GGC', 'GGG'],
        'H': ['CAT', 'CAC'],
        'I': ['ATT', 'ATC', 'ATA'],
        'K': ['AAA', 'AAG'],
        'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'],
        'M': ['ATG'],
        'N': ['AAT', 'AAC'],
        'P': ['CCT', 'CCC', 'CCA', 'CCG'],
        'Q': ['CAA', 'CAG'],
        'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
        'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],
        'T': ['ACT', 'ACC', 'ACA', 'ACG'],
        'V': ['GTT', 'GTC', 'GTA', 'GTG'],
        'W': ['TGG'],
        'Y': ['TAT', 'TAC'],
        '*': ['TAA', 'TAG', 'TGA'],
    }
    aa_list, codon_list = [], []
    for aa, codons in dic_aa_codon.items():
        for _ in codons:
            aa_list.append(aa)
        codon_list.extend(codons)
    return aa_list, codon_list


def get_codon_dict() -> dict:
    _, codon_list = get_aa_codon_lists()
    return {codon: idx for idx, codon in enumerate(codon_list, start=1)}


def get_token_aa_codon_map() -> dict:
    """Maps each AA token (1-22) → list of valid codon tokens.
    Used by the inference mask so the decoder can only predict codons
    that actually encode the amino acid at each position.
    token_aa_codon[23] = [66] maps the END token → END codon token.
    """
    aa_dict = get_aa_dict()
    aa_list, codon_list = get_aa_codon_lists()
    dic_aa_codon = {}
    for aa, token in aa_dict.items():
        if aa not in dic_aa_codon:
            dic_aa_codon[aa] = []
    for i, aa in enumerate(aa_list):
        codon_token = i + 1   # 1-indexed
        if aa in aa_dict:
            aa_token = aa_dict[aa]
            if aa_token not in dic_aa_codon:
                dic_aa_codon[aa_token] = []
            if codon_token not in dic_aa_codon.get(aa_token, []):
                dic_aa_codon.setdefault(aa_token, []).append(codon_token)

    # Rebuild cleanly keyed by integer token
    token_map: dict = {}
    for aa_char, aa_token in aa_dict.items():
        valid_codons = [
            i + 1 for i, a in enumerate(aa_list) if a == aa_char
        ]
        if valid_codons:
            token_map[aa_token] = valid_codons
    token_map[23] = [66]   # END AA token → END codon token
    return token_map


def tokenise_aa(sequences: list[str]) -> list[list[int]]:
    """Convert list of AA strings → list of integer token sequences.
    Format: [START=24, aa1, aa2, ..., aaN, END=23]
    """
    aa_dict = get_aa_dict()
    tokenised = []
    for seq in sequences:
        tokens = [24]  # START
        tokens.extend(aa_dict[ch] for ch in seq)
        tokens.append(23)  # END
        tokenised.append(tokens)
    return tokenised


def tokenise_cds(sequences: list[str]) -> list[list[int]]:
    """Convert list of nucleotide CDS strings → list of codon token sequences.
    Each 3-char codon → one integer token.
    Format: [START=65, codon1, codon2, ..., codonN, END=66]
    """
    codon_dict = get_codon_dict()
    tokenised = []
    for seq in sequences:
        tokens = [65]  # START
        n_codons = len(seq) // 3
        for i in range(n_codons):
            codon = seq[3 * i: 3 * (i + 1)]
            tokens.append(codon_dict[codon])
        tokens.append(66)  # END
        tokenised.append(tokens)
    return tokenised


def pad(sequences: list[list[int]], maxlen: int) -> np.ndarray:
    """Post-pad sequences with zeros to fixed length, truncating if longer."""
    arr = np.zeros((len(sequences), maxlen), dtype=np.int32)
    for i, seq in enumerate(sequences):
        length = min(len(seq), maxlen)
        arr[i, :length] = seq[:length]
    return arr


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw_sequences(data_dir: str = DATA_DIR) -> tuple[list, list]:
    """Load all 5 CSV files, filter sequences containing 'N', return
    (aa_sequences, cds_sequences) as flat lists."""
    aa_all, cds_all = [], []
    for fname in CSV_FILES:
        path = os.path.join(data_dir, fname)
        df = pd.read_csv(path)
        mask = ~df['CDS_Seq'].str.contains('N', na=True)
        aa_all.extend(df.loc[mask, 'AA_Seq'].tolist())
        cds_all.extend(df.loc[mask, 'CDS_Seq'].tolist())
    print(f"Loaded {len(aa_all):,} valid AA-CDS pairs from {len(CSV_FILES)} files.")
    return aa_all, cds_all


# ---------------------------------------------------------------------------
# Main preparation function
# ---------------------------------------------------------------------------

def prepare_data(
    data_dir: str = DATA_DIR,
    test_fraction: float = TEST_FRACTION,
    random_state: int = RANDOM_STATE,
    n_reps: int = N_REPS,
    aa_maxlen: int = AA_MAXLEN,
    cds_maxlen: int = CDS_MAXLEN,
    verbose: bool = True,
) -> dict:
    """
    Full data preparation pipeline. Returns a dict with keys:
        AA_tr, Cds_tr  – padded training tensors  (np.ndarray, int32)
        AA_ts, Cds_ts  – padded test tensors       (np.ndarray, int32)
        aa_dict        – amino-acid → token mapping
        codon_dict     – codon string → token mapping
        token_aa_codon – AA token → list of valid codon tokens (for inference mask)
        n_train, n_test – number of protein sequences in each split
    """
    # 1. Load raw sequences
    aa_seqs, cds_seqs = load_raw_sequences(data_dir)

    # 2. Train / test split on PROTEIN sequences (before any augmentation)
    aa_tr, aa_ts, cds_tr, cds_ts = train_test_split(
        aa_seqs, cds_seqs,
        test_size=test_fraction,
        random_state=random_state,
    )
    if verbose:
        print(f"Train proteins: {len(aa_tr):,}  |  Test proteins: {len(aa_ts):,}")

    # 3. Build augmented single-codon pairs and prepend to TRAINING data only.
    #    This teaches the model the codon table explicitly before it sees full proteins.
    aa_aug_list, codon_aug_list = get_aa_codon_lists()  # 67 entries (one per codon)
    aa_aug_rep  = aa_aug_list  * (n_reps + 1)   # 67 × 26 = 1742 single-AA strings
    cds_aug_rep = codon_aug_list * (n_reps + 1)  # 67 × 26 = 1742 single-codon strings

    aa_tr_aug  = aa_aug_rep  + aa_tr
    cds_tr_aug = cds_aug_rep + cds_tr

    # 4. Tokenise
    aa_tr_tok  = tokenise_aa(aa_tr_aug)
    cds_tr_tok = tokenise_cds(cds_tr_aug)
    aa_ts_tok  = tokenise_aa(aa_ts)
    cds_ts_tok = tokenise_cds(cds_ts)

    # 5. Pad
    AA_tr  = pad(aa_tr_tok,  aa_maxlen)
    Cds_tr = pad(cds_tr_tok, cds_maxlen)
    AA_ts  = pad(aa_ts_tok,  aa_maxlen)
    Cds_ts = pad(cds_ts_tok, cds_maxlen)

    if verbose:
        print(f"AA_tr shape : {AA_tr.shape}  (RAM ≈ {AA_tr.nbytes / 1e6:.0f} MB)")
        print(f"Cds_tr shape: {Cds_tr.shape}  (RAM ≈ {Cds_tr.nbytes / 1e6:.0f} MB)")
        print(f"AA_ts shape : {AA_ts.shape}")
        print(f"Cds_ts shape: {Cds_ts.shape}")

    return {
        "AA_tr":         AA_tr,
        "Cds_tr":        Cds_tr,
        "AA_ts":         AA_ts,
        "Cds_ts":        Cds_ts,
        "aa_dict":       get_aa_dict(),
        "codon_dict":    get_codon_dict(),
        "token_aa_codon": get_token_aa_codon_map(),
        "n_train":       len(aa_tr),
        "n_test":        len(aa_ts),
    }


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data = prepare_data(verbose=True)
    print("\ntoken_aa_codon sample (AA=1 is Ala, should map to GCT/GCC/GCA/GCG tokens 1-4):")
    print(f"  AA token 1 (A) → codon tokens: {data['token_aa_codon'][1]}")
    print(f"  AA token 11 (M) → codon tokens: {data['token_aa_codon'][11]}")
    print(f"  AA token 23 (END) → codon tokens: {data['token_aa_codon'][23]}")
