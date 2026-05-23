import numpy as np
from PIL import Image
from integral_image import compute_integral_image, rect_sum

def compute_feature(ii, kind, r, c, h, w):
    if kind == 'h2':
        return rect_sum(ii, r, c, h, w) - rect_sum(ii, r, c + w, h, w)
    elif kind == 'v2':
        return rect_sum(ii, r, c, h, w) - rect_sum(ii, r + h, c, h, w)
    elif kind == 'h3':
        left  = rect_sum(ii, r, c,         h, w)
        mid   = rect_sum(ii, r, c + w,     h, w)
        right = rect_sum(ii, r, c + 2 * w, h, w)
        return left - 2 * mid + right
    elif kind == 'd4':
        tl = rect_sum(ii, r,     c,     h, w)
        tr = rect_sum(ii, r,     c + w, h, w)
        bl = rect_sum(ii, r + h, c,     h, w)
        br = rect_sum(ii, r + h, c + w, h, w)
        return (tl + br) - (tr + bl)

def generate_features(window_size=24):
    features = []
    H = W = window_size
    for r in range(H):
        for c in range(W):
            for h in range(1, H - r + 1):
                for w in range(1, W - c + 1):
                    if c + 2 * w <= W:
                        features.append(('h2', r, c, h, w))
                    if r + 2 * h <= H:
                        features.append(('v2', r, c, h, w))
                    if c + 3 * w <= W:
                        features.append(('h3', r, c, h, w))
                    if r + 2 * h <= H and c + 2 * w <= W:
                        features.append(('d4', r, c, h, w))
    return features

# # teste
# img = np.array(Image.open("pos/55.pgm"), dtype=np.float32)
# ii = compute_integral_image(img)

# features = generate_features(24)
# print(f"Total de features geradas: {len(features)}")

# # mostra valor de algumas features na imagem
# # h3 usa 3*w colunas, então w=8 para caber em 24px
# params = {'h2': (12,12), 'v2': (12,12), 'h3': (12,8), 'd4': (12,12)}
# for kind, (h, w) in params.items():
#     val = compute_feature(ii, kind, 0, 0, h, w)
#     print(f"Feature {kind} em (0,0) {h}x{w}: {val:.1f}")

# # computa todas as features da imagem
# valores = np.array([compute_feature(ii, *f) for f in features])
# print(f"\nMaior valor: {valores.max():.1f}")
# print(f"Menor valor: {valores.min():.1f}")
# print(f"Média:       {valores.mean():.1f}")