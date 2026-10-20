import torch
import torch.nn as nn
from pathlib import Path

class AutoregressiveModel(nn.Module):
    def __init__(self, vocab_size=1024, embedding_dim=128, nhead=4, num_layers=4, max_seq_len=600):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_emb = nn.Embedding(vocab_size + 1, embedding_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, embedding_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        if isinstance(x, dict):
            x = x["indices"]

        is_3d = (x.ndim == 3)
        if is_3d:
            b, h, w = x.shape
            x_flat = x.reshape(b, -1)
        else:
            x_flat = x

        b, s = x_flat.shape
        emb = self.token_emb(x_flat) + self.pos_emb[:, :s, :]
        mask = nn.Transformer.generate_square_subsequent_mask(s, device=x.device)
        out = self.transformer(emb, mask=mask, is_causal=True)
        logits = self.head(out)

        if is_3d:
            logits = logits.reshape(b, h, w, self.vocab_size)

        return logits, {}

    @torch.no_grad()
    def generate(self, start_tokens, max_len=600, temperature=1.0, device=None):
        if device is None:
            device = next(self.parameters()).device
        
        if isinstance(start_tokens, int):
            b = start_tokens
            generated = torch.zeros((b, 1), dtype=torch.long, device=device)
        else:
            if start_tokens.ndim == 3:
                b, h, w = start_tokens.shape
                start_tokens = start_tokens.view(b, -1)
            b = start_tokens.shape[0]
            generated = start_tokens.to(device)

        for _ in range(max_len - generated.shape[1]):
            logits, _ = self.forward(generated)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
        return generated

Autoregressive = AutoregressiveModel

def load(model_path=None):
    model = AutoregressiveModel()
    if model_path is None:
        model_path = Path(__file__).parent / 'AutoregressiveModel.pth'
    if Path(model_path).exists():
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        if isinstance(state_dict, dict):
            model.load_state_dict(state_dict, strict=False)
    return model
