import numpy as np
import time
from multiprocessing import Pool, cpu_count

# globals herdados pelos workers via fork (Linux)
_X       = None
_y       = None
_pesos   = None
_indices = None  # índices ativos em X — evita copiar X_all

def avaliar_chunk(args):
    f_start, f_end = args
    melhor_erro   = float('inf')
    melhor_f      = f_start
    melhor_thresh = 0.0
    melhor_pol    = 1

    total_pos = np.sum(_pesos[_y ==  1])
    total_neg = np.sum(_pesos[_y == -1])

    for f in range(f_start, f_end):
        vals  = _X[_indices, f]   # copia só 1 coluna (~88KB), não o X todo
        ordem = np.argsort(vals)
        v_ord = vals[ordem]
        y_ord = _y[ordem]
        w_ord = _pesos[ordem]

        cum_pos = np.cumsum(w_ord * (y_ord ==  1))
        cum_neg = np.cumsum(w_ord * (y_ord == -1))

        erro_p1 = (total_pos - cum_pos) + cum_neg
        erro_p2 = cum_pos + (total_neg - cum_neg)

        if erro_p1.min() < erro_p2.min():
            idx  = np.argmin(erro_p1)
            pol  = 1
            erro = erro_p1[idx]
        else:
            idx  = np.argmin(erro_p2)
            pol  = -1
            erro = erro_p2[idx]

        if erro < melhor_erro:
            melhor_erro   = erro
            melhor_f      = f
            melhor_thresh = v_ord[idx]
            melhor_pol    = pol

    return melhor_erro, melhor_f, melhor_thresh, melhor_pol

def predict_weak(X_col, threshold, polarity):
    return np.where(polarity * X_col < polarity * threshold, 1, -1)

def treinar_adaboost(X, y, T=50, n_cores=None, indices=None):
    """
    X: matriz de features (pode ser X_all completo)
    y: labels para as amostras ativas
    indices: linhas de X a usar — se None, usa todas
    """
    global _X, _y, _pesos, _indices

    if n_cores is None:
        n_cores = cpu_count()
    if indices is None:
        indices = np.arange(len(y))

    N = len(indices)
    F = X.shape[1]
    pesos = np.ones(N) / N
    classificadores = []

    chunk_size = F // n_cores
    chunks = [(i * chunk_size, (i + 1) * chunk_size) for i in range(n_cores)]
    chunks[-1] = (chunks[-1][0], F)

    for t in range(T):
        inicio = time.time()
        pesos /= pesos.sum()

        _X, _y, _pesos, _indices = X, y, pesos, indices

        with Pool(processes=n_cores) as pool:
            resultados = pool.map(avaliar_chunk, chunks)

        melhor = min(resultados, key=lambda r: r[0])
        melhor_erro, melhor_f, melhor_thresh, melhor_pol = melhor

        alpha = 0.5 * np.log((1 - melhor_erro) / max(melhor_erro, 1e-10))
        preds = predict_weak(X[indices, int(melhor_f)], melhor_thresh, melhor_pol)
        pesos *= np.exp(-alpha * y * preds)

        acertos = np.mean(preds == y)
        tempo   = time.time() - inicio
        print(f"Round {t+1:3d}/{T} | feature={melhor_f:6d} | "
              f"erro={melhor_erro:.4f} | alpha={alpha:.4f} | "
              f"acc={acertos:.3f} | {tempo:.1f}s")

        classificadores.append((alpha, int(melhor_f), melhor_thresh, melhor_pol))

    return classificadores

def predict_adaboost(X, classificadores):
    scores = sum(
        alpha * predict_weak(X[:, int(f)], thresh, pol)
        for alpha, f, thresh, pol in classificadores
    )
    return np.where(scores >= 0, 1, -1)

if __name__ == "__main__":
    X = np.load("X.npy")
    y = np.load("y.npy")
    print(f"X={X.shape}, y={y.shape}")
    print(f"Usando {cpu_count()} núcleos\n")

    clfs = treinar_adaboost(X, y, T=20)
    np.save("classificadores.npy", clfs, allow_pickle=True)

    preds = predict_adaboost(X, clfs)
    acc   = np.mean(preds == y)
    print(f"\nAcurácia final no treino: {acc:.3f}")
