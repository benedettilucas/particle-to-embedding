from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from superpoint.models.superpoint_pytorch import SuperPoint
from accelerated_features.modules.xfeat import XFeat
from accelerated_features.modules.lighterglue import LighterGlue
import cv2
import numpy as np

def generate_random_keypoints(image_shape, num_keypoints, device="cpu", dtype=torch.float32, operation="random"):
    """
    Gera keypoints de acordo com o tipo de operação especificada.

    Args:
        image_shape: tuple (H, W) ou (B, C, H, W)
        num_keypoints: int, número de keypoints a gerar
        device: str, 'cpu' ou 'cuda'
        dtype: torch.dtype, tipo dos valores
        operation: 
            - "random": gera keypoints aleatórios uniformes (default)
            - tuple ("localized", (x, y), dispersion): gera keypoints próximos a (x, y)
            com dispersão gaussiana (desvio padrão = dispersion)

    Returns:
        torch.Tensor (B, num_keypoints, 2): coordenadas (x, y)
    """
    # Extrai H e W mesmo que venha de shape 4D
    if len(image_shape) == 4:
        _, _, H, W = image_shape
        B = image_shape[0]
    elif len(image_shape) == 2:
        H, W = image_shape
        B = 1
    else:
        raise ValueError("image_shape deve ser (H, W) ou (B, C, H, W).")

    # Caso 1: operação aleatória uniforme
    if operation == "random":
        xs = torch.rand((B, num_keypoints, 1), device=device, dtype=dtype) * (W - 1)
        ys = torch.rand((B, num_keypoints, 1), device=device, dtype=dtype) * (H - 1)

    # Caso 2: operação localizada em torno de (x, y)
    elif isinstance(operation, tuple) and len(operation) == 3 and operation[0] == "localized":
        _, center, dispersion = operation
        cx, cy = center

        # Cria ruído gaussiano ao redor do centro
        xs = torch.normal(mean=cx, std=dispersion, size=(B, num_keypoints, 1),
                        device=device, dtype=dtype)
        ys = torch.normal(mean=cy, std=dispersion, size=(B, num_keypoints, 1),
                        device=device, dtype=dtype)

        # Clampa coordenadas para ficarem dentro da imagem
        xs = xs.clamp(0, W - 1)
        ys = ys.clamp(0, H - 1)

    else:
        raise ValueError(
            "operation deve ser 'random' ou ('localized', (x, y), dispersion)"
        )

    keypoints = torch.cat([xs, ys], dim=-1)
    return keypoints

def get_descriptors_from_keypoints(model, x, feats, inference=True):
    x, rh1, rw1 = model.preprocess_tensor(x)
    B, _, H1, W1 = x.shape
    M1, K1, H1_map = model.net(x)
    M1 = F.normalize(M1, dim=1)
    keypoints = feats["keypoints"].unsqueeze(0) # add batch: (1, N, 2)
    kpts_resized = keypoints / torch.tensor(
        [rw1, rh1], device=keypoints.device
    ).view(1, 1, 2)

    features = model.interpolator(
        M1,
        kpts_resized,
        H=H1,
        W=W1
    )

    features = F.normalize(features, dim=-1)
    if inference:
        return {
            'keypoints': feats["keypoints"],
            'descriptors': features[0]
        }
    else:
        return {
            'keypoints': feats["keypoints"],
            'scores': feats["scores"],
            'descriptors': features[0]
        }

def cosine_sim(feat0, feat1):
    """
    Calcula a similaridade do cosseno entre dois vetores (1, D).

    Args:
        feat0 (torch.Tensor): tensor shape (1, D)
        feat1 (torch.Tensor): tensor shape (1, D)

    Returns:
        float: similaridade do cosseno
    """
    # remover dimensões extras -> (D,)
    v0 = feat0.squeeze()
    v1 = feat1.squeeze()

    # similaridade do cosseno
    cos = F.cosine_similarity(v0, v1, dim=0)

    return cos.item()

def plot_n_vectors(vectors, labels=None, title="Plot de múltiplos vetores 1×64"):
    """
    Plota N vetores PyTorch de shape (1, 64) no mesmo gráfico.

    Args:
        vectors (list): lista de tensores PyTorch, cada um com shape (1, 64)
        labels (list): lista opcional de rótulos. Se None, gera rótulos automáticos.
        title (str): título do gráfico
    """
    # Número de vetores
    n = len(vectors)

    # Rotulagem automática caso labels não seja fornecido
    if labels is None:
        labels = [f"vetor_{i}" for i in range(n)]

    plt.figure(figsize=(10, 5))

    for i, v in enumerate(vectors):
        # converter para numpy e remover dim extra
        v_np = v.detach().cpu().numpy().squeeze()  # vira (64,)

        plt.plot(v_np, label=labels[i], alpha=0.8)

    plt.title(title)
    plt.xlabel("Dimensão (0–63)")
    plt.ylabel("Valor")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()