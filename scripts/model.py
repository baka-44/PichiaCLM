"""
model.py
========
Pichia-CLM Arch1 — PyTorch implementation.
Exact replication of Narayanan & Love, PNAS 2026.

Architecture:
  Encoder : Bidirectional GRU (hidden=510), shared AA Embedding (dim=42)
  Decoder : Two parallel GRU heads (hidden=1020)
      • Codon head  : codon embedding(224) → GRU → dot-attention → Dense(125) → softmax(67)
      • AA aux head : shared AA embedding  → GRU → dot-attention → Dense(139) → softmax(25)
  Loss    : CrossEntropyLoss (ignore_index=0 for PAD) on both outputs
  Total parameters: 9,330,664
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Vocab / length constants
# ---------------------------------------------------------------------------
AA_VOCAB_SIZE  = 25    # 0=PAD, 1-20=AAs, 21=ambiguous, 22=stop-as-AA, 23=END, 24=START
DNA_VOCAB_SIZE = 67    # 0=PAD, 1-64=codons, 65=START, 66=END
MAX_LENGTH     = 1000
LEARNING_RATE  = 0.001
BATCH_SIZE     = 150
EPOCHS         = 100

DEFAULT_HP = dict(
    hidden_size_enc    = 510,
    embedding_size_enc = 42,    # AA embedding dimension (shared enc + AA aux dec)
    embedding_size_dec = 224,   # codon embedding dimension
    dense_layer_size   = 125,   # codon decoder dense
    dense_layer_size_aa = 139,  # AA aux decoder dense
    drop_rate          = 0.0,
    drop_rate_aa       = 0.7,
)


class PichiaCLMArch1(nn.Module):
    """
    Encoder-Decoder GRU model for codon optimisation.

    Parameter count breakdown (hp = DEFAULT_HP):
      AA Embedding  (25 × 42)          :     1,050
      Codon Emb     (67 × 224)         :    15,008
      BiGRU Encoder (I=42, H=510×2)    : 1,695,240
      Codon Dec GRU (I=224, H=1020)    : 3,812,760
      AA Dec GRU    (I=42,  H=1020)    : 3,255,840
      Dense codon   (2040→125→67)      :   263,567
      Dense AA      (2040→139→25)      :   287,199
      ─────────────────────────────────────────────
      Total                            : 9,330,664
    """

    def __init__(self, hp=None):
        super().__init__()
        if hp is None:
            hp = DEFAULT_HP

        hidden   = hp['hidden_size_enc']        # 510
        enc_emb  = hp['embedding_size_enc']     # 42
        dec_emb  = hp['embedding_size_dec']     # 224
        dense    = hp['dense_layer_size']       # 125
        dense_aa = hp['dense_layer_size_aa']    # 139
        drop     = hp['drop_rate']              # 0.0
        drop_aa  = hp['drop_rate_aa']           # 0.7

        # ── Shared AA embedding (encoder + AA aux decoder) ───────────────
        self.aa_embedding    = nn.Embedding(AA_VOCAB_SIZE,  enc_emb, padding_idx=0)
        self.codon_embedding = nn.Embedding(DNA_VOCAB_SIZE, dec_emb, padding_idx=0)

        # ── Bidirectional GRU encoder ────────────────────────────────────
        self.encoder_gru = nn.GRU(
            enc_emb, hidden,
            batch_first=True, bidirectional=True,
        )

        # ── Codon decoder GRU ────────────────────────────────────────────
        # Initial state = BiGRU fwd+bwd final states concatenated = 2*hidden
        self.decoder_codon_gru = nn.GRU(dec_emb, 2 * hidden, batch_first=True)

        # ── AA auxiliary decoder GRU (shares AA embedding) ───────────────
        self.decoder_aa_gru = nn.GRU(enc_emb, 2 * hidden, batch_first=True)

        # ── Codon head: cat(dec_out[1020], attn[1020]) = 4*hidden = 2040 ─
        self.dense_codon   = nn.Linear(4 * hidden, dense)
        self.dropout_codon = nn.Dropout(drop)
        self.output_codon  = nn.Linear(dense, DNA_VOCAB_SIZE)

        # ── AA aux head ──────────────────────────────────────────────────
        self.dense_aa   = nn.Linear(4 * hidden, dense_aa)
        self.dropout_aa = nn.Dropout(drop_aa)
        self.output_aa  = nn.Linear(dense_aa, AA_VOCAB_SIZE)

    # ── Attention ────────────────────────────────────────────────────────────
    @staticmethod
    def dot_attention(query, keys):
        """
        Unscaled dot-product attention — matches Keras Attention layer default.
        query : (B, T_q, H)
        keys  : (B, T_k, H)   [key = value]
        → context (B, T_q, H)
        """
        scores  = torch.bmm(query, keys.transpose(1, 2))  # (B, T_q, T_k)
        weights = F.softmax(scores, dim=-1)
        return torch.bmm(weights, keys)                    # (B, T_q, H)

    # ── Forward pass ─────────────────────────────────────────────────────────
    def forward(self, enc_aa, dec_codon, dec_aa):
        """
        enc_aa    : (B, 1000)  encoder AA tokens
        dec_codon : (B, 999)   decoder codon tokens (teacher forcing)
        dec_aa    : (B, 1000)  decoder AA tokens (auxiliary head)

        Returns
        -------
        codon_logits : (B, 999,  67)
        aa_logits    : (B, 1000, 25)
        """
        # ── Encoder ──────────────────────────────────────────────────────
        enc_emb_out = self.aa_embedding(enc_aa)               # (B, 1000, 42)
        enc_seq, enc_state = self.encoder_gru(enc_emb_out)
        # enc_seq   : (B, 1000, 1020)
        # enc_state : (2, B, 510)   [fwd_final, bwd_final]

        # Concatenate fwd + bwd final states → decoder initial hidden state
        enc_final = torch.cat([enc_state[0], enc_state[1]], dim=-1)  # (B, 1020)
        h0 = enc_final.unsqueeze(0)                                    # (1, B, 1020)

        # ── Codon decoder ─────────────────────────────────────────────────
        dec_emb_out = self.codon_embedding(dec_codon)         # (B, 999, 224)
        dec_seq, _  = self.decoder_codon_gru(dec_emb_out, h0) # (B, 999, 1020)

        attn_c   = self.dot_attention(dec_seq, enc_seq)        # (B, 999, 1020)
        cat_c    = torch.cat([dec_seq, attn_c], dim=-1)        # (B, 999, 2040)

        out_c    = torch.tanh(self.dense_codon(cat_c))         # (B, 999, 125)
        out_c    = self.dropout_codon(out_c)
        codon_logits = self.output_codon(out_c)                 # (B, 999, 67)

        # ── AA auxiliary decoder ──────────────────────────────────────────
        aa_emb_out = self.aa_embedding(dec_aa)                 # (B, 1000, 42) shared
        aa_seq, _  = self.decoder_aa_gru(aa_emb_out, h0)      # (B, 1000, 1020)

        attn_aa  = self.dot_attention(aa_seq, enc_seq)         # (B, 1000, 1020)
        cat_aa   = torch.cat([aa_seq, attn_aa], dim=-1)        # (B, 1000, 2040)

        out_aa   = torch.tanh(self.dense_aa(cat_aa))           # (B, 1000, 139)
        out_aa   = self.dropout_aa(out_aa)
        aa_logits = self.output_aa(out_aa)                      # (B, 1000, 25)

        return codon_logits, aa_logits


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

def build_training_model(hp=None, device=None):
    """Instantiate PichiaCLMArch1 and move to device."""
    model = PichiaCLMArch1(hp)
    if device is not None:
        model = model.to(device)
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = build_training_model()
    total = count_parameters(model)
    print(f'Total parameters: {total:,}')
    # Expected: 9,330,664
