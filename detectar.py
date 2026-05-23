import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import pickle
import numpy as np
from threading import Thread

from integral_image import compute_integral_image
from haar_features import generate_features

cap = cv2.VideoCapture(0)
cv2.imshow("Viola-Jones", np.zeros((480, 640, 3), dtype=np.uint8))
cv2.waitKey(1)
print("Carregando modelo e features...")

WINDOW = 24
features = generate_features(WINDOW)

with open("cascade.pkl", "rb") as f:
    cascade = pickle.load(f)

# extrai só as features usadas pelo cascade (evita lookup de 134k)
feats_usadas = {}
for clfs, _ in cascade:
    for alpha, f_idx, thresh_w, pol in clfs:
        i = int(f_idx)
        if i not in feats_usadas:
            feats_usadas[i] = features[i]

print(f"Cascade: {len(cascade)} estágios | Features usadas: {len(feats_usadas)}")
print("Pressione 'q' para sair.")

# --- sliding window vetorizado ---

def rect_sum_vec(ii, rows, cols, h, w):
    H, W = ii.shape
    r2 = np.clip(rows + h - 1, 0, H - 1)
    c2 = np.clip(cols + w - 1, 0, W - 1)
    total = ii[r2, c2].copy()
    mr = rows > 0
    mc = cols > 0
    total[mr]      -= ii[rows[mr] - 1, c2[mr]]
    total[mc]      -= ii[r2[mc], cols[mc] - 1]
    total[mr & mc] += ii[rows[mr & mc] - 1, cols[mr & mc] - 1]
    return total.astype(np.float32)

def feature_vec(ii, kind, rows, cols, h, w):
    if kind == 'h2':
        return rect_sum_vec(ii, rows, cols, h, w) - rect_sum_vec(ii, rows, cols+w, h, w)
    elif kind == 'v2':
        return rect_sum_vec(ii, rows, cols, h, w) - rect_sum_vec(ii, rows+h, cols, h, w)
    elif kind == 'h3':
        return (rect_sum_vec(ii, rows, cols, h, w)
                - 2*rect_sum_vec(ii, rows, cols+w, h, w)
                + rect_sum_vec(ii, rows, cols+2*w, h, w))
    else:  # d4
        return ((rect_sum_vec(ii, rows, cols, h, w)
                 + rect_sum_vec(ii, rows+h, cols+w, h, w))
                - (rect_sum_vec(ii, rows, cols+w, h, w)
                   + rect_sum_vec(ii, rows+h, cols, h, w)))

def detectar_frame(gray, passo=8, escala=1.25):
    ii = compute_integral_image(gray.astype(np.float32))
    H, W = gray.shape
    candidatos = []

    tamanho = WINDOW
    while tamanho <= min(H, W):
        s = tamanho / WINDOW

        ys = np.arange(0, H - tamanho, passo)
        xs = np.arange(0, W - tamanho, passo)
        yy, xx = np.meshgrid(ys, xs, indexing='ij')
        rows = yy.ravel()
        cols = xx.ravel()
        vivos = np.ones(len(rows), dtype=bool)

        for clfs, threshold in cascade:
            if not vivos.any():
                break
            r = rows[vivos]
            c = cols[vivos]
            scores = np.zeros(len(r), dtype=np.float32)

            for alpha, f_idx, thresh_w, pol in clfs:
                kind, fr, fc, fh, fw = feats_usadas[int(f_idx)]
                # escala as coordenadas da feature pro tamanho atual
                fr_s = int(fr * s)
                fc_s = int(fc * s)
                fh_s = max(1, int(fh * s))
                fw_s = max(1, int(fw * s))
                vals  = feature_vec(ii, kind, r + fr_s, c + fc_s, fh_s, fw_s)
                scores += alpha * np.where(pol * vals < pol * thresh_w, 1.0, -1.0)

            vivos[vivos] = scores >= threshold

        for wy, wx in zip(rows[vivos], cols[vivos]):
            candidatos.append((int(wx), int(wy), tamanho, tamanho))

        tamanho = int(tamanho * escala)

    return candidatos

import time, json, pathlib

rostos_atual = []
detectando   = False
log          = []  # guarda histórico de detecções
frames_save  = []  # frames anotados para salvar

def rodar_deteccao(gray, sx, sy, t):
    global rostos_atual, detectando
    t_ini = time.time()
    encontrados = detectar_frame(gray)

    if len(encontrados) > 0:
        rects = encontrados + encontrados
        grupos, _ = cv2.groupRectangles(rects, groupThreshold=2, eps=0.3)
        encontrados = grupos.tolist() if len(grupos) > 0 else []

    rostos_atual = [(int(x*sx), int(y*sy), int(w*sx), int(h*sy))
                    for x, y, w, h in encontrados]
    log.append({
        "t":         round(t, 2),
        "duracao_s": round(time.time() - t_ini, 2),
        "n_rostos":  len(rostos_atual),
        "rostos":    rostos_atual
    })
    detectando = False

DURACAO = 5
t_start = time.time()
print(f"Rodando por {DURACAO} segundos...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    t_agora = time.time() - t_start

    for (x, y, w, h) in rostos_atual:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(frame, f"{t_agora:.1f}s | Rostos: {len(rostos_atual)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Viola-Jones", frame)
    frames_save.append(frame.copy())

    if not detectando:
        detectando = True
        small = cv2.resize(frame, (320, 240))
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        sx = frame.shape[1] / 320
        sy = frame.shape[0] / 240
        Thread(target=rodar_deteccao, args=(gray, sx, sy, t_agora),
               daemon=True).start()

    if t_agora >= DURACAO or cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# salva resultados
out = pathlib.Path("deteccao_resultado")
out.mkdir(exist_ok=True)

# salva log JSON
with open(out / "log.json", "w") as f:
    json.dump(log, f, indent=2)

# salva 1 frame por segundo (evita salvar 150 frames)
fps_aprox = len(frames_save) / DURACAO
for i, frame in enumerate(frames_save):
    if i % max(1, int(fps_aprox)) == 0:
        cv2.imwrite(str(out / f"frame_{i:04d}.jpg"), frame)

print(f"\nSalvo em deteccao_resultado/")
print(f"  {len(log)} detecções realizadas")
print(f"  {len([f for f in out.glob('*.jpg')])} frames salvos")
if log:
    tempos = [e['duracao_s'] for e in log]
    print(f"  Tempo médio por detecção: {sum(tempos)/len(tempos):.2f}s")
    print(f"  Rostos detectados por ciclo: {[e['n_rostos'] for e in log]}")
