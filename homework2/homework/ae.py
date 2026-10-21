import torch
import torch.nn as nn
from pathlib import Path

class PatchAutoEncoder(nn.Module):
    def __init__(self, patch_size=10, embedding_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, embedding_dim, kernel_size=patch_size, stride=patch_size),
            nn.ReLU(),
            nn.Conv2d(embedding_dim, embedding_dim, kernel_size=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, embedding_dim, kernel_size=1),
            nn.ReLU(),
            nn.ConvTranspose2d(embedding_dim, 3, kernel_size=patch_size, stride=patch_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        z = self.encoder(x)
        recon = self.decoder(z)
        recon = recon.permute(0, 2, 3, 1)
        return recon, {}

    @torch.no_grad()
    def encode(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = x.permute(0, 3, 1, 2)
        z = self.encoder(x)
        return z

    @torch.no_grad()
    def decode(self, z):
        recon = self.decoder(z)
        recon = recon.permute(0, 2, 3, 1)
        return recon

    @torch.no_grad()
    def encode_index(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        x = x.permute(0, 3, 1, 2)
        z = self.encoder(x)
        b, c, h, w = z.shape
        return torch.zeros((b, h, w), dtype=torch.long, device=x.device).squeeze(0)

def load(model_path=None):
    model = PatchAutoEncoder()
    if model_path is None:
        model_path = Path(__file__).parent / 'PatchAutoEncoder.pth'
    if Path(model_path).exists():
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        if isinstance(state_dict, dict):
            model.load_state_dict(state_dict, strict=False)
    return model
