"""
backend/load_compat.py
------------------
Fixes cross-version Keras model loading.

Colab uses Keras >= 3.4 which serialises a 'quantization_config' key
into Dense layer configs. Older local Keras builds reject this key.
We monkey-patch Dense.from_config once, before any model is loaded,
so the key is silently dropped.
"""

import tensorflow as tf


def _patch_dense():
    """Patch keras.layers.Dense.from_config to ignore unknown keys."""
    import keras  # keras is bundled with TF 2.x

    try:
        _orig_from_config = keras.layers.Dense.from_config.__func__
    except AttributeError:
        # Already a regular classmethod — get it differently
        _orig_from_config = staticmethod(keras.layers.Dense.from_config).__func__

    @classmethod
    def _compat_from_config(cls, config):
        config.pop("quantization_config", None)
        return _orig_from_config(cls, config)

    keras.layers.Dense.from_config = _compat_from_config
    tf.keras.layers.Dense.from_config = _compat_from_config


# ── Apply the patch immediately on import ─────────────────────────────────────
_patch_dense()


def load_model_compat(path: str) -> tf.keras.Model:
    """
    Load a .keras model file with cross-version compatibility.
    Handles 'quantization_config' introduced in Keras 3.4+.
    """
    _patch_dense()   # idempotent — safe to call multiple times
    return tf.keras.models.load_model(path, compile=False)
