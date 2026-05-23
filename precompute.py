import numpy as np
from pathlib import Path
from PIL import Image
from multiprocessing import Pool, cpu_count
import time

from integral_image import compute_integral_image
from haar_features import generate_features, compute_feature

WINDOW = 24
features = generate_features(WINDOW)

def processar_imagem(path):
    img = np.array(Image.open(path), dtype=np.float32)
    ii  = compute_integral_image(img)
    return np.array([compute_feature(ii, kind, r, c, h, w)
                     for kind, r, c, h, w in features], dtype=np.float32)

def load_paths(folder, limit=None):
    paths = sorted(Path(folder).glob("*.pgm"))
    return paths[:limit] if limit else paths

if __name__ == "__main__":
    pos_paths = load_paths("pos", limit=1000)
    neg_paths = load_paths("neg", limit=1000)
    all_paths = pos_paths + neg_paths
    labels    = np.array([1] * len(pos_paths) + [-1] * len(neg_paths))

    n_cores = cpu_count()
    print(f"Usando {n_cores} núcleos")
    print(f"Total de imagens: {len(all_paths)}")
    print(f"Features por imagem: {len(features)}")
    print("Pré-computando...")

    inicio = time.time()
    with Pool(processes=n_cores) as pool:
        resultados = pool.map(processar_imagem, all_paths)

    X = np.array(resultados, dtype=np.float32)

    np.save("X.npy", X)
    np.save("y.npy", labels)

    tempo = time.time() - inicio
    print(f"\nConcluído em {tempo:.1f}s")
    print(f"X.shape={X.shape}, y.shape={labels.shape}")
