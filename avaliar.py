import numpy as np
from pathlib import Path
from PIL import Image
from multiprocessing import Pool, cpu_count

from integral_image import compute_integral_image
from haar_features import generate_features, compute_feature
from adaboost import predict_adaboost

WINDOW  = 24
features = generate_features(WINDOW)

def processar_imagem(path):
    img = np.array(Image.open(path), dtype=np.float32)
    ii  = compute_integral_image(img)
    return np.array([compute_feature(ii, kind, r, c, h, w)
                    for kind, r, c, h, w in features], dtype=np.float32)

if __name__ == "__main__":
    # carrega imagens que NÃO foram usadas no treino (offset de 1000)
    pos_paths = sorted(Path("pos").glob("*.pgm"))[1000:2000]
    neg_paths = sorted(Path("neg").glob("*.pgm"))[1000:]

    all_paths = pos_paths + neg_paths
    y_test    = np.array([1] * len(pos_paths) + [-1] * len(neg_paths))
    print(f"Teste: {len(pos_paths)} faces, {len(neg_paths)} não-faces")

    print("Computando features de teste...")
    with Pool(cpu_count()) as pool:
        resultados = pool.map(processar_imagem, all_paths)
    X_test = np.array(resultados, dtype=np.float32)

    clfs   = np.load("classificadores.npy", allow_pickle=True)
    preds  = predict_adaboost(X_test, clfs)

    tp = np.sum((preds ==  1) & (y_test ==  1))
    tn = np.sum((preds == -1) & (y_test == -1))
    fp = np.sum((preds ==  1) & (y_test == -1))
    fn = np.sum((preds == -1) & (y_test ==  1))

    acc       = (tp + tn) / len(y_test)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nResultados no conjunto de TESTE:")
    print(f"  Acurácia:  {acc:.3f}")
    print(f"  Precision: {precision:.3f}  (de tudo que classifiquei como face, quantas eram faces)")
    print(f"  Recall:    {recall:.3f}  (de todas as faces reais, quantas encontrei)")
    print(f"  F1-score:  {f1:.3f}")
    print(f"\n  TP={tp}  TN={tn}  FP={fp}  FN={fn}")