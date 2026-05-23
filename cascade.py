import numpy as np
import pickle
from pathlib import Path
from PIL import Image
from multiprocessing import Pool, cpu_count

from integral_image import compute_integral_image
from haar_features import generate_features, compute_feature
from adaboost import treinar_adaboost, predict_weak

WINDOW   = 24
features = generate_features(WINDOW)

def processar_imagem(path):
    img = np.array(Image.open(path), dtype=np.float32)
    ii  = compute_integral_image(img)
    return np.array([compute_feature(ii, kind, r, c, h, w)
                     for kind, r, c, h, w in features], dtype=np.float32)

def scores_adaboost(X, indices, clfs):
    return sum(
        alpha * predict_weak(X[indices, int(f)], thresh, pol)
        for alpha, f, thresh, pol in clfs
    )

def ajustar_threshold(scores, y, recall_alvo=0.99):
    faces_scores = np.sort(scores[y == 1])
    idx = int((1 - recall_alvo) * len(faces_scores))
    return faces_scores[idx]

def predict_cascade(X, cascade):
    vivos = np.ones(len(X), dtype=bool)
    for clfs, threshold in cascade:
        idx = np.where(vivos)[0]
        s   = scores_adaboost(X, idx, clfs)
        vivos[idx[s < threshold]] = False
    return np.where(vivos, 1, -1)

if __name__ == "__main__":
    pos_paths = sorted(Path("pos").glob("*.pgm"))
    neg_paths = sorted(Path("neg").glob("*.pgm"))
    all_paths = pos_paths + neg_paths

    n_pos = len(pos_paths)
    n_neg = len(neg_paths)
    F     = len(features)
    gb    = len(all_paths) * F * 4 / 1e9
    print(f"Positivos: {n_pos} | Negativos: {n_neg}")
    print(f"X_all em memória: {gb:.1f} GB")

    X_all_path = Path("X_all.npy")
    if X_all_path.exists():
        print("Carregando X_all do disco...")
        X_all = np.load(X_all_path)
        print(f"  Carregado: {X_all.shape}")
    else:
        print("Pré-computando todas as features em batches...")
        X_all = np.empty((len(all_paths), F), dtype=np.float32)
        BATCH = 500
        for start in range(0, len(all_paths), BATCH):
            batch = all_paths[start:start + BATCH]
            with Pool(cpu_count()) as pool:
                res = pool.map(processar_imagem, batch)
            X_all[start:start + len(batch)] = res
            del res
            print(f"  {min(start + BATCH, len(all_paths))}/{len(all_paths)}")
        np.save(X_all_path, X_all)
        print("  X_all salvo em X_all.npy")

    # índices globais — sem copiar dados
    pos_idx    = np.arange(n_pos)
    neg_idx    = np.arange(n_pos, n_pos + n_neg)
    neg_ativos = neg_idx.copy()

    config = [
        {'rounds': 10, 'recall': 0.99},
        {'rounds': 20, 'recall': 0.99},
        {'rounds': 30, 'recall': 0.99},
    ]

    cascade = []

    for i, cfg in enumerate(config):
        print(f"\n=== Estágio {i+1} ===")
        n_ativos = len(pos_idx) + len(neg_ativos)
        print(f"  Positivos: {len(pos_idx)} | Negativos ativos: {len(neg_ativos)}")

        # índices e labels das amostras ativas — sem copiar X_all
        indices_ativos = np.concatenate([pos_idx, neg_ativos])
        y_train        = np.array([1] * len(pos_idx) + [-1] * len(neg_ativos))

        # passa X_all inteiro + índices → adaboost nunca copia X_all
        clfs = treinar_adaboost(X_all, y_train, T=cfg['rounds'],
                                indices=indices_ativos)

        # scores nas amostras ativas
        scores_pos = scores_adaboost(X_all, pos_idx,    clfs)
        scores_neg = scores_adaboost(X_all, neg_ativos, clfs)
        scores_all = np.concatenate([scores_pos, scores_neg])

        threshold = ajustar_threshold(scores_all, y_train, recall_alvo=cfg['recall'])

        preds  = np.where(scores_all >= threshold, 1, -1)
        tp     = np.sum((preds ==  1) & (y_train ==  1))
        fp     = np.sum((preds ==  1) & (y_train == -1))
        fn     = np.sum((preds == -1) & (y_train ==  1))
        recall = tp / (tp + fn)
        print(f"  Threshold={threshold:.2f} | Recall={recall:.3f} | FP={fp}")

        sobrevivem = neg_ativos[scores_neg >= threshold]
        print(f"  Negativos rejeitados: {len(neg_ativos) - len(sobrevivem)}")
        neg_ativos = sobrevivem

        cascade.append((clfs, threshold))

        if len(neg_ativos) == 0:
            print("  Todos os negativos rejeitados — cascade completo!")
            break

    with open("cascade.pkl", "wb") as f:
        pickle.dump(cascade, f)
    print("\nCascade salvo em cascade.pkl")

    # avaliação final
    with open("cascade.pkl", "rb") as f:
        cascade = pickle.load(f)
    test_idx = np.concatenate([pos_idx[:500], neg_idx[:2000]])
    y_test   = np.array([1] * 500 + [-1] * 2000)
    preds    = predict_cascade(X_all[test_idx], cascade)

    tp = np.sum((preds ==  1) & (y_test ==  1))
    fp = np.sum((preds ==  1) & (y_test == -1))
    fn = np.sum((preds == -1) & (y_test ==  1))
    tn = np.sum((preds == -1) & (y_test == -1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nResultados finais do cascade:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1-score:  {f1:.3f}")
    print(f"  TP={tp} TN={tn} FP={fp} FN={fn}")
