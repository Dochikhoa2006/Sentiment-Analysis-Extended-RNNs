"""TensorFlow model construction."""

from __future__ import annotations

from typing import Any

from sentiment_analyzer.config import ModelConfig


def build_model(config: ModelConfig) -> Any:
    """Build and compile a bidirectional GRU or LSTM classifier."""

    try:
        import tensorflow as tf
        from tensorflow.keras import Input, Model
        from tensorflow.keras.layers import GRU, LSTM, Bidirectional, Dense, Masking
    except ImportError as exc:
        raise RuntimeError("Install ML dependencies with: pip install -e '.[ml]'") from exc

    tf.keras.utils.set_random_seed(config.seed)
    recurrent_layer = GRU if config.architecture == "gru" else LSTM

    inputs = Input(
        shape=(config.sequence_length, config.embedding_dimension),
        name="review_embeddings",
    )
    hidden = Masking(mask_value=0.0, name="padding_mask")(inputs)
    hidden = Bidirectional(
        recurrent_layer(config.recurrent_units[0], return_sequences=True),
        name=f"bidirectional_{config.architecture}_1",
    )(hidden)
    hidden = Bidirectional(
        recurrent_layer(config.recurrent_units[1]),
        name=f"bidirectional_{config.architecture}_2",
    )(hidden)
    outputs = Dense(config.number_of_classes, activation="softmax", name="sentiment")(hidden)

    model = Model(inputs=inputs, outputs=outputs, name=f"app_review_bi{config.architecture}")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
