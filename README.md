# Viola-Jones Face Detector — From Scratch with NumPy

Implementação completa do detector de faces de Viola & Jones (2001) em Python puro com NumPy, sem frameworks de deep learning. Inclui treino do zero, detecção em tempo real via webcam e uma derivação matemática completa de cada componente.

---

## Índice

1. [Imagem Integral](#1-imagem-integral)
2. [Features de Haar](#2-features-de-haar)
3. [Classificador Fraco — Decision Stump](#3-classificador-fraco--decision-stump)
4. [AdaBoost — Derivação Completa](#4-adaboost--derivação-completa)
5. [Cascade de Classificadores](#5-cascade-de-classificadores)
6. [Complexidade e Análise Assintótica](#6-complexidade-e-análise-assintótica)
7. [Resultados](#7-resultados)
8. [Como Usar](#8-como-usar)

---

## 1. Imagem Integral

### 1.1 Definição Formal

Seja $I : \Omega \to \mathbb{R}$ uma imagem em escala de cinza definida sobre o domínio discreto $\Omega = \{0,\ldots,H-1\} \times \{0,\ldots,W-1\}$. A **imagem integral** (ou *summed area table*) é o operador $\mathcal{S} : \mathbb{R}^{H \times W} \to \mathbb{R}^{H \times W}$ definido por:

$$\mathcal{S}[I](r, c) = \sum_{r'=0}^{r} \sum_{c'=0}^{c} I(r', c')$$

Equivalentemente, como composição de dois operadores de soma cumulativa ao longo de eixos ortogonais:

$$\mathcal{S} = \mathcal{C}_{\text{col}} \circ \mathcal{C}_{\text{row}}$$

onde:

$$
(\mathcal{C}_{\text{row}} A)(r,c) = \sum_{c'=0}^{c} A(r,c'), \qquad (\mathcal{C}_{\text{col}} A)(r,c) = \sum_{r'=0}^{r} A(r',c)
$$

### 1.2 Teorema da Soma Retangular em O(1)

**Teorema.** Seja $\text{II} = \mathcal{S}[I]$. A soma de todos os pixels no retângulo $R = [r_1, r_2] \times [c_1, c_2]$ é dada por:

$$\sum_{r=r_1}^{r_2} \sum_{c=c_1}^{c_2} I(r,c) = \text{II}(r_2, c_2) - \text{II}(r_1-1, c_2) - \text{II}(r_2, c_1-1) + \text{II}(r_1-1, c_1-1)$$

**Prova.** Pela inclusão-exclusão sobre os quatro quadrantes da decomposição da imagem integral:

$$\text{II}(r_2, c_2) = A + B + C + D$$
$$\text{II}(r_1-1, c_2) = A + B$$
$$\text{II}(r_2, c_1-1) = A + C$$
$$\text{II}(r_1-1, c_1-1) = A$$

onde $A$, $B$, $C$, $D$ são as somas dos quatro blocos na decomposição. Portanto:

$$D = \text{II}(r_2,c_2) - \text{II}(r_1-1,c_2) - \text{II}(r_2,c_1-1) + \text{II}(r_1-1,c_1-1) \qquad \blacksquare$$

A imagem integral é pré-computada em $O(HW)$ e cada consulta retangular subsequente é $O(1)$, independente do tamanho do retângulo. Sem ela, computar $K$ features sobre $N$ janelas de tamanho $w \times w$ custaria $O(NKw^2)$; com ela, $O(HW + NK)$.

---

## 2. Features de Haar

### 2.1 Definição como Funcionais Lineares

Uma **feature de Haar** é um funcional $\phi_j : \mathbb{R}^{H \times W} \to \mathbb{R}$ da forma:

$$\phi_j(I) = \sum_{k} s_k \cdot \text{RectSum}(I, R_k^{(j)})$$

onde $s_k \in \{-1, +1\}$ são sinais associados a cada retângulo $R_k^{(j)}$ da feature $j$. Os quatro tipos implementados são:

| Tipo | Expressão |
|------|-----------|
| `h2` (2 retângulos horizontais) | $\text{RS}(r,c,h,w) - \text{RS}(r,c+w,h,w)$ |
| `v2` (2 retângulos verticais) | $\text{RS}(r,c,h,w) - \text{RS}(r+h,c,h,w)$ |
| `h3` (3 retângulos horizontais) | $\text{RS}(r,c,h,w) - 2\cdot\text{RS}(r,c+w,h,w) + \text{RS}(r,c+2w,h,w)$ |
| `d4` (4 retângulos em diagonal) | $(RS_{TL} + RS_{BR}) - (RS_{TR} + RS_{BL})$ |

### 2.2 Espaço de Features

Para uma janela de $24 \times 24$ pixels, o número total de features distintas é:

$$|\mathcal{F}| = \sum_{\text{tipo}} \sum_{r,c,h,w} \mathbf{1}[\text{feature válida}] = 134{,}736$$

Este espaço é exponencialmente maior que a dimensão da imagem ($24^2 = 576$), capturando estruturas de múltiplas escalas simultaneamente. A feature `h2` captura bordas verticais (contraste testa/olhos); `v2` captura bordas horizontais; `h3` captura estruturas de banda (nariz); `d4` captura cruzamentos (cantos dos olhos).

---

## 3. Classificador Fraco — Decision Stump

### 3.1 Formulação

Dado um conjunto de amostras $\{(x_i, y_i)\}_{i=1}^N$ com $y_i \in \{-1, +1\}$ e pesos $w_i > 0$ com $\sum_i w_i = 1$, um **decision stump** para a feature $j$ é:

$$h_j(x; \theta, p) = \begin{cases} +1 & \text{se } p \cdot \phi_j(x) < p \cdot \theta \\ -1 & \text{caso contrário} \end{cases}$$

onde $\theta \in \mathbb{R}$ é o threshold e $p \in \{-1, +1\}$ é a polaridade.

### 3.2 Minimização do Erro Ponderado

O erro ponderado é:

$$\varepsilon_j(\theta, p) = \sum_{i=1}^{N} w_i \cdot \mathbf{1}[h_j(x_i; \theta, p) \neq y_i]$$

Para encontrar $(\theta^*, p^*)$ ótimos, ordena-se as amostras por $\phi_j(x_i)$: seja $\sigma$ a permutação tal que $\phi_j(x_{\sigma(1)}) \leq \cdots \leq \phi_j(x_{\sigma(N)})$. Definem-se os pesos acumulados:

$$T_+(i) = \sum_{k=1}^{i} w_{\sigma(k)} \cdot \mathbf{1}[y_{\sigma(k)} = +1], \qquad T_-(i) = \sum_{k=1}^{i} w_{\sigma(k)} \cdot \mathbf{1}[y_{\sigma(k)} = -1]$$

e os totais $S_+ = T_+(N)$, $S_- = T_-(N)$. Então:

$$\varepsilon(\theta_i, p=+1) = T_+(i) + (S_- - T_-(i))$$
$$\varepsilon(\theta_i, p=-1) = T_-(i) + (S_+ - T_+(i))$$

O mínimo global é obtido varrendo todos os $N$ thresholds em $O(N \log N)$ (dominado pela ordenação).

---

## 4. AdaBoost — Derivação Completa

### 4.1 Modelagem como Minimização de Perda Exponencial

AdaBoost resolve o problema de **Stagewise Additive Modeling** sob a perda exponencial. O classificador final é:

$$F(x) = \sum_{t=1}^{T} \alpha_t h_t(x)$$

A função de perda exponencial sobre o conjunto de treino é:

$$\mathcal{L}(F) = \sum_{i=1}^{N} \exp(-y_i F(x_i))$$

**Proposição.** A minimização *greedy* de $\mathcal{L}$ por adição de um classificador fraco $h_t$ com peso $\alpha_t$ é equivalente ao algoritmo AdaBoost de Freund & Schapire (1997).

**Prova.** Após $t-1$ iterações, $F_{t-1}(x_i)$ está fixo. Minimizamos sobre $(h_t, \alpha_t)$:

$$\mathcal{L}(F_{t-1} + \alpha h) = \sum_i \exp(-y_i F_{t-1}(x_i)) \cdot \exp(-y_i \alpha h(x_i))$$

Definindo $w_i^{(t)} = \exp(-y_i F_{t-1}(x_i))$ (não normalizado), temos:

$$\mathcal{L} = \sum_i w_i^{(t)} e^{-\alpha y_i h(x_i)}$$

Separando corretos ($y_i h(x_i) = 1$) e incorretos ($y_i h(x_i) = -1$):

$$\mathcal{L} = e^{-\alpha} \sum_{y_i = h(x_i)} w_i^{(t)} + e^{\alpha} \sum_{y_i \neq h(x_i)} w_i^{(t)}$$

Seja $\varepsilon_t = \frac{\sum_{y_i \neq h(x_i)} w_i^{(t)}}{\sum_i w_i^{(t)}}$ o erro ponderado normalizado. Então:

$$\mathcal{L} = W^{(t)} \left[ e^{-\alpha}(1 - \varepsilon_t) + e^{\alpha} \varepsilon_t \right]$$

Minimizando em $\alpha$ via $\partial \mathcal{L}/\partial \alpha = 0$:

$$-e^{-\alpha}(1-\varepsilon_t) + e^{\alpha}\varepsilon_t = 0 \implies e^{2\alpha} = \frac{1-\varepsilon_t}{\varepsilon_t}$$

$$\boxed{\alpha_t = \frac{1}{2} \ln \frac{1 - \varepsilon_t}{\varepsilon_t}}$$

Para encontrar $h_t$ ótimo, basta minimizar $\varepsilon_t$, pois $\mathcal{L}$ é monotonamente decrescente em $\varepsilon_t$ para $\alpha = \alpha_t(\varepsilon_t)$.

### 4.2 Atualização de Pesos

Substituindo $F_t = F_{t-1} + \alpha_t h_t$:

$$w_i^{(t+1)} = \exp(-y_i F_t(x_i)) = w_i^{(t)} \exp(-\alpha_t y_i h_t(x_i))$$

Após normalização por $Z_t = \sum_i w_i^{(t+1)}$:

$$\tilde{w}_i^{(t+1)} = \frac{w_i^{(t+1)}}{Z_t} = \frac{w_i^{(t)} e^{-\alpha_t y_i h_t(x_i)}}{Z_t}$$

O fator de normalização tem forma fechada:

$$Z_t = W^{(t)} \cdot 2\sqrt{\varepsilon_t(1-\varepsilon_t)}$$

### 4.3 Teorema de Convergência do Erro de Treino

**Teorema (Freund & Schapire, 1997).** O erro empírico de $F_T$ sobre o conjunto de treino satisfaz:

$$\frac{1}{N}\sum_{i=1}^{N} \mathbf{1}[F_T(x_i) \leq 0] \leq \prod_{t=1}^{T} Z_t = \prod_{t=1}^{T} 2\sqrt{\varepsilon_t(1-\varepsilon_t)}$$

**Prova.** Por indução, $\frac{1}{N}\sum_i \mathbf{1}[y_i F_T(x_i) \leq 0] \leq \frac{1}{N}\sum_i e^{-y_i F_T(x_i)}$. Como $w_i^{(1)} = 1/N$ e $w_i^{(T+1)} = \frac{1}{N} \prod_{t=1}^T Z_t \cdot e^{-y_i F_T(x_i)} / \prod Z_t$, somando sobre $i$ e usando $\sum_i w_i^{(T+1)} = 1$ obtemos o resultado. $\blacksquare$

**Corolário.** Se $\varepsilon_t \leq \frac{1}{2} - \gamma$ para algum $\gamma > 0$ (condição de aprendizabilidade fraca), então:

$$\text{Erro}(F_T) \leq \prod_{t=1}^T 2\sqrt{(1/2-\gamma)(1/2+\gamma)} = \prod_{t=1}^T \sqrt{1-4\gamma^2} \leq e^{-2\gamma^2 T}$$

O erro de treino cai **exponencialmente** em $T$.

### 4.4 Interpretação como Gradient Boosting

AdaBoost é equivalente ao Gradient Boosting com perda exponencial. O gradiente negativo é:

$$-\frac{\partial \mathcal{L}}{\partial F(x_i)} = y_i \exp(-y_i F(x_i)) = y_i w_i^{(t)}$$

O classificador fraco $h_t$ é o minimizador do erro ponderado, que corresponde a ajustar $h_t$ aos pseudo-resíduos $r_i = y_i w_i^{(t)}$.

### 4.5 Margem e Generalização

O **margin** da amostra $i$ é definido como:

$$\rho_i = \frac{y_i \sum_t \alpha_t h_t(x_i)}{\sum_t \alpha_t}$$

Schapire et al. (1998) provam que, com probabilidade $\geq 1-\delta$ sobre uma amostra de tamanho $N$:

$$\Pr_{D}[y F(x) \leq 0] \leq \Pr_S[\rho_i \leq \theta] + O\!\left(\sqrt{\frac{d \log^2(N/d)}{\theta^2 N} + \frac{\log(1/\delta)}{N}}\right)$$

onde $d$ é a dimensão VC dos classificadores fracos. Para decision stumps com $K$ features, $d = O(\log K)$.

---

## 5. Cascade de Classificadores

### 5.1 Formulação como Decisão Sequencial

O cascade é uma cadeia de $S$ classificadores $\{(F_s, \theta_s)\}_{s=1}^S$ dispostos em série. A decisão final é:

$$C(x) = +1 \iff \forall s \in \{1,\ldots,S\} : F_s(x) \geq \theta_s$$

### 5.2 Análise da Taxa de Falsos Positivos

Seja $f_s$ a taxa de falsos positivos do estágio $s$ (probabilidade de uma não-face passar pelo estágio $s$). A taxa de falsos positivos do cascade completo é:

$$F = \prod_{s=1}^{S} f_s$$

Seja $d_s$ a taxa de detecção (recall) do estágio $s$. A taxa de detecção total é:

$$D = \prod_{s=1}^{S} d_s$$

**Proposição.** Para atingir $D \geq D_{\min}$ e $F \leq F_{\max}$, se todos os estágios têm $d_s = d$ e $f_s = f$, então o número de estágios necessário é:

$$S = \left\lceil \frac{\log F_{\max}}{\log f} \right\rceil \quad \text{com} \quad d \geq D_{\min}^{1/S}$$

### 5.3 Ajuste de Threshold por Recall

Para cada estágio $s$, o threshold $\theta_s$ não é o padrão $0$ do AdaBoost, mas sim ajustado para garantir $d_s \geq d_{\min}$ (aqui $d_{\min} = 0.99$):

$$\theta_s = \text{Quantil}_{1 - d_{\min}}\left(\{F_s(x_i) : y_i = +1\}\right)$$

Isto é, $\theta_s$ é escolhido como o $(1 - d_{\min})$-quantil dos scores das faces verdadeiras no conjunto de treino.

### 5.4 Hard Negative Mining

Em cada estágio $s > 1$, o conjunto de negativos é os **falsos positivos** dos estágios anteriores:

$$\mathcal{N}_s = \{x \in \mathcal{N} : \forall s' < s, F_{s'}(x) \geq \theta_{s'}\}$$

Isso implementa um regime de bootstrap: cada estágio aprende a rejeitar exatamente os negativos mais difíceis — aqueles que iludiram todos os estágios anteriores. Esta estratégia é formalmente análoga ao **Sequential Probability Ratio Test (SPRT)** de Wald (1947), onde cada estágio atualiza uma estatística de razão de verossimilhança.

### 5.5 Eficiência Computacional

O speedup do cascade sobre um detector monolítico é:

$$\text{Speedup} = \frac{1}{\sum_{s=1}^{S} C_s \prod_{s'=1}^{s-1} f_{s'}}$$

onde $C_s$ é o custo computacional do estágio $s$. Para $f_s \approx 0.1$ e $S = 3$:

$$\text{Speedup} \approx \frac{1}{C_1 + 0.1 C_2 + 0.01 C_3} \gg 1$$

Sobre imagens naturais onde $\geq 99.9\%$ das janelas não contêm rosto, o cascade processa menos de $0.1\%$ das janelas com o classificador completo.

---

## 6. Complexidade e Análise Assintótica

| Etapa | Complexidade |
|-------|-------------|
| Pré-computo da imagem integral | $O(HW)$ |
| Avaliação de uma feature (sem II) | $O(w^2)$ |
| Avaliação de uma feature (com II) | $O(1)$ |
| Treino de um weak classifier ($N$ amostras) | $O(N \log N)$ |
| Uma rodada de AdaBoost ($K$ features) | $O(NK \log N)$ |
| Treino completo ($T$ rounds) | $O(TNK \log N)$ |
| Detecção vetorizada (todas janelas, 1 feature) | $O(HW/p^2)$ com passo $p$ |
| Detecção completa (cascade com $M$ features) | $O(MHW/p^2)$ amortizado |

A vetorização do sliding window elimina o loop Python sobre janelas: para cada feature, computamos $\text{RectSum}$ simultaneamente para todas as $\lfloor H/p \rfloor \times \lfloor W/p \rfloor$ janelas via operações numpy sobre arrays de índices.

---

## 7. Resultados

### 7.1 Dataset

- **Positivos**: 16.520 imagens de faces (MIT CBCL Face Database, 24×24 px)
- **Negativos**: 6.038 patches sem face (24×24 px)

### 7.2 Avaliação no Conjunto de Teste (holdout)

| Métrica | Valor |
|---------|-------|
| Acurácia | 96.2% |
| Precision | 82.7% |
| Recall | 97.8% |
| F1-score | 89.6% |

### 7.3 Cascade (3 estágios)

| Estágio | Rounds | Negativos Rejeitados | FP Restantes |
|---------|--------|---------------------|--------------|
| 1 | 10 | 5.610 / 6.038 (92.9%) | 428 |
| 2 | 20 | 311 / 428 (72.7%) | 117 |
| 3 | 30 | 117 / 117 (100%) | 0 |

**Avaliação final do cascade**: Precision=1.000, Recall=0.974, F1=0.987

### 7.4 Detecção em Tempo Real

- Tempo médio por ciclo de detecção: **44ms** (~22 FPS de detecção)
- Frame de câmera: 640×480, reduzido para 320×240 para detecção
- Sliding window vetorizado com NumPy: todas as posições processadas em paralelo por feature

---

## 8. Como Usar

### Instalação

```bash
git clone https://github.com/Mathweuzz/viola-jones-from-scratch
cd viola-jones-from-scratch
pip install numpy pillow opencv-python scikit-learn
```

### Pré-computar features

```bash
python precompute.py       # 1000 amostras rápido
# ou
python cascade.py          # dataset completo (~12GB RAM)
```

### Treinar AdaBoost

```bash
python adaboost.py
```

### Treinar Cascade

```bash
python cascade.py
```

### Avaliar

```bash
python avaliar.py
```

### Detecção via webcam

```bash
python detectar.py
```

---

## Referências

1. Viola, P., & Jones, M. (2001). *Rapid object detection using a boosted cascade of simple features*. CVPR.
2. Freund, Y., & Schapire, R. E. (1997). *A decision-theoretic generalization of on-line learning and an application to boosting*. JCSS, 55(1).
3. Schapire, R. E., Freund, Y., Bartlett, P., & Lee, W. S. (1998). *Boosting the margin: A new explanation for the effectiveness of voting methods*. Annals of Statistics.
4. Friedman, J., Hastie, T., & Tibshirani, R. (2000). *Additive logistic regression: a statistical view of boosting*. Annals of Statistics.
5. Wald, A. (1947). *Sequential Analysis*. Wiley.
