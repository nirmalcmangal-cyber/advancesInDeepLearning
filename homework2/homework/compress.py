import torch

class Compressor:
    def __init__(self, bsq_ae, ar_model):
        self.bsq_ae = bsq_ae
        self.ar_model = ar_model

    def compress(self, x):
        # x is [H, W, C]
        if x.ndim == 3:
            x = x.unsqueeze(0)
        # x is [1, H, W, 3]
        _, indices_dict = self.bsq_ae(x)
        indices = indices_dict["indices"]
        return indices.squeeze(0).cpu().tolist()

    def decompress(self, compressed_data, h=None, w=None):
        device = next(self.bsq_ae.parameters()).device
        indices = torch.tensor(compressed_data, device=device).unsqueeze(0) # [1, h_p, w_p]

        # Map indices back to the BSQ binary codes
        codes = self.bsq_ae.bsq._index_to_code(indices) # [1, h_p, w_p, 10]
        
        # Project codes up to matching embedding dimensions
        embeddings = self.bsq_ae.bsq.proj_up(codes)     # [1, h_p, w_p, 64]

        # Permute to channels-first before decoding
        embeddings = embeddings.permute(0, 3, 1, 2)     # [1, 64, h_p, w_p]
        recon = self.bsq_ae.ae.decoder(embeddings)       # [1, 3, H, W]

        # Convert back to channels-last [H, W, 3]
        return recon.squeeze(0).permute(1, 2, 0)
