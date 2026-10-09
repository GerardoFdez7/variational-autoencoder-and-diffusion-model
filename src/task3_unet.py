"""Task 3.1: U-Net pequena condicional (tiempo + clase) que predice el ruido epsilon."""
import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

T = 1000
NUM_CLASSES = 10
NULL_CLASS = 10          # la clase 10 es la condicion nula (para guia sin clasificador)
EMB_DIM = 128            # dimension d de los embeddings de tiempo y clase


def sinusoidal_embedding(t, dim):
    """Positional encoding de la Semana 6, evaluado en la posicion t:

        PE(t, 2i)   = sin(t / 10000^(2i/d))
        PE(t, 2i+1) = cos(t / 10000^(2i/d))

    t: tensor (B,) con instantes (1..T). Devuelve (B, dim) con senos en las
    posiciones pares y cosenos en las impares.
    """
    i = torch.arange(dim // 2, device=t.device, dtype=torch.float32)
    freqs = torch.exp(-math.log(10000.0) * (2 * i) / dim)        # 1 / 10000^(2i/d)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)            # (B, d/2)
    pe = torch.zeros(t.shape[0], dim, device=t.device)
    pe[:, 0::2] = torch.sin(args)
    pe[:, 1::2] = torch.cos(args)
    return pe


class ConvBlock(nn.Module):
    """Conv-GN-SiLU, inyeccion del embedding (t + clase) por broadcasting, Conv-GN-SiLU."""

    def __init__(self, c_in, c_out, emb_dim, groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, c_out)
        self.emb_proj = nn.Linear(emb_dim, c_out)                 # (B,d) -> (B,c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, c_out)
        self.act = nn.SiLU()

    def forward(self, x, emb):
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.emb_proj(emb)[:, :, None, None]              # broadcast sobre H x W
        return self.act(self.norm2(self.conv2(h)))


class UNet(nn.Module):
    """U-Net con dos niveles de resolucion:

        28x28, 32 canales  ->  14x14, 64 canales  ->  cuello de botella 14x14, 64 canales
        -> subida a 14x14 (concat skip) -> subida a 28x28 (concat skip) -> conv 1x1 a 1 canal
    """

    def __init__(self, emb_dim=EMB_DIM, num_classes=NUM_CLASSES + 1):
        super().__init__()
        self.emb_dim = emb_dim
        # MLP sobre el embedding sinusoidal del instante t.
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        # Embedding de clase: 0..9 clases reales, 10 = condicion nula.
        self.class_emb = nn.Embedding(num_classes, emb_dim)

        self.enc1 = ConvBlock(1, 32, emb_dim)                     # 28x28, 32 canales
        self.down = nn.Conv2d(32, 32, 3, stride=2, padding=1)     # 28 -> 14
        self.enc2 = ConvBlock(32, 64, emb_dim)                    # 14x14, 64 canales
        self.bottleneck = ConvBlock(64, 64, emb_dim)              # 14x14, cuello de botella
        self.dec2 = ConvBlock(64 + 64, 64, emb_dim)               # concat con salto de enc2
        self.up = nn.ConvTranspose2d(64, 32, 2, stride=2)         # 14 -> 28
        self.dec1 = ConvBlock(32 + 32, 32, emb_dim)               # concat con salto de enc1
        self.out = nn.Conv2d(32, 1, 1)                            # convolucion final a 1 canal

    def forward(self, x, t, c):
        """x: (B,1,28,28), t: (B,) en 1..T, c: (B,) en 0..10. Devuelve eps_hat (B,1,28,28)."""
        emb = self.time_mlp(sinusoidal_embedding(t, self.emb_dim)) + self.class_emb(c)

        s1 = self.enc1(x, emb)                                    # (B,32,28,28)
        s2 = self.enc2(self.down(s1), emb)                        # (B,64,14,14)
        b = self.bottleneck(s2, emb)                              # (B,64,14,14)
        d2 = self.dec2(torch.cat([b, s2], dim=1), emb)            # (B,64,14,14)
        d1 = self.dec1(torch.cat([self.up(d2), s1], dim=1), emb)  # (B,32,28,28)
        return self.out(d1)                                       # (B,1,28,28)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    common.set_seed()
    model = UNet()
    total = count_params(model)
    por_modulo = {name: count_params(m) for name, m in model.named_children()}

    # Verificacion de forma con un lote de prueba.
    x = torch.randn(4, 1, 28, 28)
    t = torch.randint(1, T + 1, (4,))
    c = torch.tensor([0, 3, 9, NULL_CLASS])
    out = model(x, t, c)
    assert out.shape == x.shape, f"forma inesperada: {tuple(out.shape)}"

    emb = sinusoidal_embedding(t, EMB_DIM)
    results = {"seed": common.SEED, "emb_dim": EMB_DIM, "parametros_totales": total,
               "parametros_por_modulo": por_modulo,
               "forma_entrada": list(x.shape), "forma_salida": list(out.shape),
               "forma_embedding_sinusoidal": list(emb.shape)}
    path = common.save_results("task3_1_results.json", results)

    print("Parametros totales:", total)
    for k, v in por_modulo.items():
        print(f"  {k}: {v}")
    print("Entrada:", tuple(x.shape), "-> salida:", tuple(out.shape))
    print("JSON:", path)


if __name__ == "__main__":
    main()
