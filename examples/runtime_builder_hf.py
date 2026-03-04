"""
Example runtime builder for model_tools runtime tracing.

Usage:
  python3 analyze_model_architecture.py \
    --mode runtime \
    --runtime-builder examples/runtime_builder_hf.py:build_runtime_case \
    --output-dir analysis_runtime
"""

from __future__ import annotations

import inspect

import torch
from transformers import BertConfig, BertModel


def build_runtime_case():
    config = BertConfig(
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_hidden_layers=2,
        vocab_size=30_522,
    )
    model = BertModel(config).eval()
    batch_size = 2
    seq_len = 16
    inputs = {
        "input_ids": torch.randint(0, config.vocab_size, (batch_size, seq_len), dtype=torch.long),
        "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.long),
        "token_type_ids": torch.zeros((batch_size, seq_len), dtype=torch.long),
    }
    return {
        "model": model,
        "inputs": inputs,
        "source_path": inspect.getsourcefile(model.__class__),
        "entry_class": model.__class__.__name__,
        "metadata": {"example": "bert_runtime_dummy"},
    }
