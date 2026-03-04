import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel


class ToyAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states):
        query_states = self.query(hidden_states)
        key_states = self.key(hidden_states)
        scores = torch.matmul(query_states, key_states.transpose(-1, -2))
        probs = F.softmax(scores, dim=-1)
        context = torch.matmul(probs, self.value(hidden_states))
        return context


class ToyLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = ToyAttention(config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states):
        attention_output = self.attention(hidden_states)
        layer_output = self.output(attention_output)
        return layer_output


class ToyEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([ToyLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states):
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
        return hidden_states


class ToyModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.encoder = ToyEncoder(config)
        self.norm = nn.LayerNorm(config.hidden_size)

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.embeddings(input_ids)
        hidden_states = self.encoder(hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states
