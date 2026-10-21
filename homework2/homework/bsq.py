import torch
import torch.nn as nn
from pathlib import Path
from homework.ae import PatchAutoEncoder

def diff_sign(x):
    return x + (torch.sign(x) - x).detach()

class BSQ(nn.Module):
    def __init__(self, codebook_bits=10, embedding_dim=64):
        super().__init__()
        self.codebook_bits = codebook_bits
        self.embedding_dim = embedding_dim
        self.proj_down = nn.Linear(embedding_dim, codebook_bits)
        self.proj_up = nn.Linear(codebook_bits, embedding_dim)

    def _code_to_index(self, code):
        binary_code = (code + 1) / 2
        powers = 2 ** torch.arange(self.codebook_bits, device=code.device)
        return (binary_code * powers).sum(dim=-1).long()

    def _index_to_code(self, index):
        powers = 2 ** torch.arange(self.codebook_bits, device=index.device)
        code = ((index.unsqueeze(-1) // powers) % 2).float()
        return code * 2 - 1

    def forward(self, x):
        z = self.proj_down(x)
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
        q = diff_sign(z)
        indices = self._code_to_index(q)
        recon = self.proj_up(q)
        return recon, indices

class BSQPatchAutoEncoder(nn.Module):
    def __init__(self, patch_size=5, embedding_dim=64, codebook_bits=10):
        super().__init__()
        self.ae = PatchAutoEncoder(patch_size, embedding_dim)
        self.bsq = BSQ(codebook_bits, embedding_dim)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        z = self.ae.encoder(x)
        z = z.permute(0, 2, 3, 1)
        quant, indices = self.bsq(z)
        quant = quant.permute(0, 3, 1, 2)
        recon = self.ae.decoder(quant)
        recon = recon.permute(0, 2, 3, 1)
        return recon, {"indices": indices}

    @torch.no_grad()
    def encode_index(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = x.permute(0, 3, 1, 2)
        z = self.ae.encoder(x)
        z = z.permute(0, 2, 3, 1)
        _, indices = self.bsq(z)
        return indices

    @torch.no_grad()
    def decode_index(self, indices):
        if indices.ndim == 2:
            indices = indices.unsqueeze(0)
        codes = self.bsq._index_to_code(indices)
        embeddings = self.bsq.proj_up(codes)
        embeddings = embeddings.permute(0, 3, 1, 2)
        recon = self.ae.decoder(embeddings)
        recon = recon.permute(0, 2, 3, 1)
        return recon

class Tokenizer:
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def encode_index(self, x):
        return self.model.encode_index(x)

    @torch.no_grad()
    def decode_index(self, x):
        return self.model.decode_index(x)

    def to(self, device):
        if hasattr(self.model, 'to'):
            self.model = self.model.to(device)
        return self

def load(model_path=None):
    if model_path is None:
        model_path = Path(__file__).parent / 'BSQPatchAutoEncoder.pth'
    model_path = Path(model_path)
    model = BSQPatchAutoEncoder()
    if model_path.exists():
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        if isinstance(state_dict, dict):
            if any(k.startswith("encoder") or k.startswith("decoder") for k in state_dict.keys()):
                fixed_state_dict = {}
                for k, v in state_dict.items():
                    fixed_state_dict[f"ae.{k}"] = v
                model.load_state_dict(fixed_state_dict, strict=False)
            else:
                model.load_state_dict(state_dict, strict=False)
    return model
