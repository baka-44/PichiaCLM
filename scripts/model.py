"""
model.py
========
Pichia-CLM Arch1 — exact replication of the final architecture from
Narayanan & Love, PNAS 2026.

Architecture (from ApplyingModel_Arch1.ipynb, Setting_no=1, Round3.csv):
  - Encoder : Bidirectional GRU (hidden=510), shared AA Embedding (dim=42)
  - Decoder : Two parallel GRU heads (size=2×510=1020)
      • Codon head  : codon embedding (dim=224) → GRU → Attention → Dense(125) → softmax(67)
      • AA aux head : shared AA embedding        → GRU → Attention → Dense(139) → softmax(25)
  - Loss    : sparse_categorical_crossentropy on both outputs (multi-task)
  - Inputs  : 3  (encoder AA, decoder codon teacher-forcing, decoder AA auxiliary)
  - Outputs : 2  (predicted codons, reconstructed AA)

The dual decoder with AA reconstruction is a multi-task regulariser —
the model must simultaneously predict the right codon AND reconstruct
the amino acid at each position. This steers the shared encoder
embeddings towards biologically meaningful representations.

Final hyperparameters (Arch1, Row 1 of BO Round3.csv):
  Enc hidden size   : 510
  Enc Embedding dim : 42    (AA embedding, shared between encoder + AA aux decoder)
  Dec Embedding dim : 224   (codon embedding)
  Dense layer size  : 125   (codon decoder head)
  Dense layer size aa: 139  (AA aux decoder head)
  Drop rate         : 0.0   (codon path)
  Drop rate aa      : 0.7   (AA aux path)
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, GRU,
    Concatenate, Attention, TimeDistributed,
    Dense, Dropout,
)
from tensorflow.keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy

# ---------------------------------------------------------------------------
# Default hyperparameters — Arch1 final (Setting_no=1, Round3.csv)
# ---------------------------------------------------------------------------
DEFAULT_HP = dict(
    hidden_size_enc   = 510,
    embedding_size_enc = 42,   # AA embedding dimension (shared)
    embedding_size_dec = 224,  # codon embedding dimension
    dense_layer_size   = 125,  # codon decoder dense
    dense_layer_size_aa = 139, # AA aux decoder dense
    drop_rate          = 0.0,
    drop_rate_aa       = 0.7,
)

# Fixed vocab/length constants
AA_VOCAB_SIZE  = 25
DNA_VOCAB_SIZE = 67
MAX_LENGTH     = 1000   # model sequence length (encoder + decoder)
LEARNING_RATE  = 0.001
BATCH_SIZE     = 150
EPOCHS         = 100


def build_training_model(hp: dict = None) -> Model:
    """
    Build the full encoder-decoder training model with teacher forcing.

    Three inputs:
      input_sequence   : (batch, MAX_LENGTH)     – encoder AA tokens (positions 1:1001 of padded)
      decoder_inputs   : (batch, MAX_LENGTH-1)   – decoder codon tokens, teacher-forced (positions 0:999)
      decoder_inputs_aa: (batch, MAX_LENGTH)     – decoder AA tokens for aux head (positions 0:1000)

    Two outputs (both use sparse_categorical_crossentropy):
      logits    : (batch, MAX_LENGTH-1, DNA_VOCAB_SIZE) – codon probabilities
      logits_aa : (batch, MAX_LENGTH,   AA_VOCAB_SIZE)  – AA reconstruction probabilities
    """
    if hp is None:
        hp = DEFAULT_HP

    hidden  = hp['hidden_size_enc']
    enc_emb = hp['embedding_size_enc']
    dec_emb = hp['embedding_size_dec']
    dense   = hp['dense_layer_size']
    dense_aa = hp['dense_layer_size_aa']
    drop    = hp['drop_rate']
    drop_aa = hp['drop_rate_aa']

    # ---- Encoder --------------------------------------------------------
    input_sequence = Input(shape=(MAX_LENGTH,), name='encoder_aa_input')

    # Shared AA embedding — reused by the AA auxiliary decoder head
    encod_emb = Embedding(
        input_dim=AA_VOCAB_SIZE, output_dim=enc_emb,
        trainable=True, mask_zero=True, name='aa_embedding'
    )
    embedding = encod_emb(input_sequence)

    encoder = Bidirectional(
        GRU(hidden, return_sequences=True, return_state=True),
        merge_mode='concat', name='bidir_gru_encoder'
    )
    encoder_sequence, encoder_final_f, encoder_final_b = encoder(embedding)
    encoder_final = Concatenate(axis=-1, name='encoder_final_state')(
        [encoder_final_f, encoder_final_b]
    )   # shape: (batch, 2*hidden)

    # ---- Codon decoder head -------------------------------------------
    decoder_inputs = Input(shape=(MAX_LENGTH - 1,), name='decoder_codon_input')

    dex = Embedding(
        input_dim=DNA_VOCAB_SIZE, output_dim=dec_emb,
        trainable=True, mask_zero=True, name='codon_embedding'
    )
    final_dex = dex(decoder_inputs)

    decoder = GRU(2 * hidden, return_sequences=True, return_state=True,
                  name='gru_decoder_codon')
    decoder_sequence, _ = decoder(final_dex, initial_state=encoder_final)

    attn_layer = Attention(name='attention_codon')
    attn_out = attn_layer([decoder_sequence, encoder_sequence])

    decoder_concat = Concatenate(axis=-1, name='codon_concat')(
        [decoder_sequence, attn_out]
    )

    intermediate = TimeDistributed(Dense(dense, activation='tanh'), name='dense_codon')
    intermediate_out = intermediate(decoder_concat)

    dropout_layer = Dropout(drop, name='dropout_codon')
    dropout_out = dropout_layer(intermediate_out)

    dense_layer = TimeDistributed(Dense(DNA_VOCAB_SIZE, activation='softmax'),
                                  name='output_codon')
    logits = dense_layer(dropout_out)

    # ---- AA auxiliary decoder head ------------------------------------
    decoder_inputs_aa = Input(shape=(MAX_LENGTH,), name='decoder_aa_input')

    # Reuse the same AA embedding as the encoder
    final_dex_aa = encod_emb(decoder_inputs_aa)

    decoder_aa = GRU(2 * hidden, return_sequences=True, return_state=True,
                     name='gru_decoder_aa')
    decoder_sequence_aa, _ = decoder_aa(final_dex_aa, initial_state=encoder_final)

    attn_layer_aa = Attention(name='attention_aa')
    attn_out_aa = attn_layer_aa([decoder_sequence_aa, encoder_sequence])

    decoder_concat_aa = Concatenate(axis=-1, name='aa_concat')(
        [decoder_sequence_aa, attn_out_aa]
    )

    intermediate_aa = TimeDistributed(Dense(dense_aa, activation='tanh'), name='dense_aa')
    intermediate_out_aa = intermediate_aa(decoder_concat_aa)

    dropout_layer_aa = Dropout(drop_aa, name='dropout_aa')
    dropout_out_aa = dropout_layer_aa(intermediate_out_aa)

    dense_layer_aa = TimeDistributed(Dense(AA_VOCAB_SIZE, activation='softmax'),
                                     name='output_aa')
    logits_aa = dense_layer_aa(dropout_out_aa)

    # ---- Compile -------------------------------------------------------
    model = Model(
        inputs=[input_sequence, decoder_inputs, decoder_inputs_aa],
        outputs=[logits, logits_aa],
        name='PichiaCLM_Arch1_training',
    )
    # Keras 3 requires one metrics entry per output when using a list
    model.compile(
        loss=sparse_categorical_crossentropy,
        optimizer=Adam(learning_rate=LEARNING_RATE),
        metrics=[['accuracy'], ['accuracy']],
    )
    return model


def build_inference_models(training_model: Model, hp: dict = None):
    """
    Decompose the trained model into separate encoder and decoder
    models for autoregressive inference (one codon at a time).

    Returns:
        encoder_model   : AA sequence → (encoder_final_state, encoder_sequences)
        decoder_model   : (codon_token_t, state_t, encoder_sequences) → (codon_probs, state_t+1)
    """
    if hp is None:
        hp = DEFAULT_HP

    hidden = hp['hidden_size_enc']

    # Retrieve trained layers by name
    aa_emb_layer     = training_model.get_layer('aa_embedding')
    encoder_layer    = training_model.get_layer('bidir_gru_encoder')
    codon_emb_layer  = training_model.get_layer('codon_embedding')
    decoder_layer    = training_model.get_layer('gru_decoder_codon')
    attn_layer       = training_model.get_layer('attention_codon')
    dense_inter      = training_model.get_layer('dense_codon')
    dropout_layer    = training_model.get_layer('dropout_codon')
    dense_out        = training_model.get_layer('output_codon')

    # --- Encoder model -------------------------------------------------
    enc_input = Input(shape=(MAX_LENGTH,), name='enc_inf_aa_input')
    enc_emb   = aa_emb_layer(enc_input)
    enc_seq, enc_f, enc_b = encoder_layer(enc_emb)
    enc_final = Concatenate(axis=-1)([enc_f, enc_b])

    encoder_model = Model(enc_input, [enc_final, enc_seq],
                          name='encoder_inference')

    # --- Decoder model (one step at a time) ----------------------------
    dec_state_input  = Input(shape=(2 * hidden,),          name='dec_inf_state')
    enc_seq_input    = Input(shape=(MAX_LENGTH, 2 * hidden), name='dec_inf_enc_seq')
    dec_token_input  = Input(shape=(1,),                   name='dec_inf_token')

    dec_emb = codon_emb_layer(dec_token_input)
    dec_out, dec_state = decoder_layer(dec_emb, initial_state=dec_state_input)

    attn_out_inf = attn_layer([dec_out, enc_seq_input])
    dec_concat   = Concatenate(axis=-1)([dec_out, attn_out_inf])

    inter_out  = dense_inter(dec_concat)
    drop_out   = dropout_layer(inter_out)   # dropout is off at inference (training=False)
    codon_prob = dense_out(drop_out)

    decoder_model = Model(
        inputs=[dec_token_input, dec_state_input, enc_seq_input],
        outputs=[codon_prob, dec_state],
        name='decoder_inference',
    )

    return encoder_model, decoder_model


if __name__ == "__main__":
    model = build_training_model()
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")
