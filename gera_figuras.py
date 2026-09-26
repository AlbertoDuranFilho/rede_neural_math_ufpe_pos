import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import numpy as np

PASTA = 'figuras'
FUNDO = '#fcfcfb'
TEXTO = '#0b0b0b'
TEXTO_2 = '#52514e'
LINHA = '#8a8984'
AZUL = '#2a78d6'       # tanh (camadas escondidas)
LARANJA = '#eb6834'    # sigmoide (saída)
GRAD = '#b3261e'       # gradientes (backward)


def _ponta(p, q, r):
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    return p + d / np.linalg.norm(d) * r


def _rotulos(ax, p, q, cima, baixo, t=0.5, tam=7.5):
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = p + (q - p) * t
    d = q - p
    ang = np.degrees(np.arctan2(d[1], d[0]))
    if ang > 90:
        ang -= 180
    if ang < -90:
        ang += 180
    n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    if n[1] < 0:
        n = -n
    if cima:
        ax.text(*(m + n * 0.17), cima, rotation=ang, ha='center', va='center',
                fontsize=tam, color=TEXTO_2, rotation_mode='anchor')
    if baixo:
        ax.text(*(m - n * 0.19), baixo, rotation=ang, ha='center',
                va='center', fontsize=tam - 0.5, color=GRAD,
                rotation_mode='anchor')


def seta(ax, p, q, rp=0.0, rq=0.0, cima='', baixo='', t=0.5):
    a = _ponta(p, q, rp) if rp else np.asarray(p, float)
    b = _ponta(q, p, rq) if rq else np.asarray(q, float)
    ax.annotate('', xy=b, xytext=a,
                arrowprops=dict(arrowstyle='-|>', color=LINHA, lw=1,
                                shrinkA=0, shrinkB=0, mutation_scale=9))
    _rotulos(ax, a, b, cima, baixo, t)


# ---------------------------------------------------------- arquitetura
def arquitetura():
    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.patch.set_facecolor(FUNDO)
    ax.set_facecolor(FUNDO)

    R = 0.42
    ent = {'x[0]': (0, 3.2), 'x[1]': (0, 1.2)}
    c1 = {0: (3, 4.2), 1: (3, 2.2), 2: (3, 0.2)}
    c2 = {3: (6, 3.2), 4: (6, 1.2)}
    sai = {5: (9, 2.2)}

    def neuronio(pos, texto, cor, bias):
        ax.add_patch(Circle(pos, R, facecolor=FUNDO, edgecolor=cor, lw=2,
                            zorder=3))
        ax.text(pos[0], pos[1] + 0.07, texto, ha='center', va='center',
                fontsize=10, color=TEXTO, zorder=4)
        if bias:
            ax.text(pos[0], pos[1] - 0.17, bias, ha='center', va='center',
                    fontsize=7, color=TEXTO_2, zorder=4)

    def liga(p, q, rotulo, t=0.3):
        a, b = _ponta(p, q, R), _ponta(q, p, R)
        ax.plot([a[0], b[0]], [a[1], b[1]], color=LINHA, lw=0.9, zorder=1)
        _rotulos(ax, a, b, rotulo, '', t=t, tam=7)

    for i, (nome, p) in enumerate(ent.items()):
        for j, q in c1.items():
            liga(p, q, f'w0[{j},{i}]', t=0.28 if i == 0 else 0.22)
    for k, q in c2.items():
        for j, p in c1.items():
            liga(p, q, f'w1[{k - 3},{j}]', t=0.32)
    for k, p in c2.items():
        liga(p, sai[5], f'w2[{k - 3}]', t=0.4)

    for nome, p in ent.items():
        neuronio(p, nome, TEXTO_2, None)
    for j, p in c1.items():
        neuronio(p, f'y{j}', AZUL, f'b0[{j}]')
    for k, p in c2.items():
        neuronio(p, f'y{k}', AZUL, f'b1[{k - 3}]')
    neuronio(sai[5], 'y5', LARANJA, 'b2[0]')

    for x, t1, t2, cor in ((0, 'entrada', '2 atributos', TEXTO_2),
                           (3, '1ª camada escondida', '3 neurônios · tanh', AZUL),
                           (6, '2ª camada escondida', '2 neurônios · tanh', AZUL),
                           (9, 'saída', '1 neurônio · sigmoide', LARANJA)):
        ax.text(x, 5.25, t1, ha='center', fontsize=10, color=TEXTO,
                weight='bold')
        ax.text(x, 4.95, t2, ha='center', fontsize=9, color=cor)

    ax.text(9.55, 2.2, 'e = y5 − d\nL = ½ e²', ha='left', va='center',
            fontsize=9, color=TEXTO_2)
    ax.set_xlim(-0.8, 11)
    ax.set_ylim(-0.6, 5.6)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA, 'arquitetura.png'), dpi=160,
                facecolor=FUNDO)
    plt.close(fig)


# grafo computacional
def grafo():
    fig, ax = plt.subplots(figsize=(28, 13.5))
    fig.patch.set_facecolor(FUNDO)
    ax.set_facecolor(FUNDO)
    R = 0.3            
    BW, BH = 0.95, 0.42 
    RB = 0.5        

    def op(pos, s):
        ax.add_patch(Circle(pos, R, facecolor=FUNDO, edgecolor=AZUL, lw=1.4,
                            zorder=3))
        ax.text(*pos, s, ha='center', va='center', fontsize=13, color=TEXTO,
                zorder=4)

    def caixa(pos, s, cor):
        ax.add_patch(FancyBboxPatch((pos[0] - BW / 2, pos[1] - BH / 2), BW,
                                    BH, boxstyle='round,pad=0.02',
                                    facecolor=FUNDO, edgecolor=cor, lw=1.6,
                                    zorder=3))
        ax.text(*pos, s, ha='center', va='center', fontsize=10, color=TEXTO,
                zorder=4)

    def entrada(q, cima, baixo, dx=-1.1, dy=0.45):
        p = (q[0] + dx, q[1] + dy)
        seta(ax, p, q, 0, R, cima, baixo, t=0.45)

    # 1ª camada escondida 
    Y1 = {0: 11.0, 1: 7.0, 2: 3.0}
    X_MUL1, X_SOM1, X_BIAS1, X_ACT1, X_RAMO1 = 1.6, 3.2, 4.6, 6.6, 7.9
    ramo = {}
    for j, yc in Y1.items():
        m0, m1 = (X_MUL1, yc + 0.8), (X_MUL1, yc - 0.8)
        sm, bs, at = (X_SOM1, yc), (X_BIAS1, yc), (X_ACT1, yc)
        op(m0, '×'); op(m1, '×'); op(sm, '+'); op(bs, '+')
        caixa(at, 'tanh', AZUL)
        entrada(m0, f'w0[{j},0]', f'grad_s{j}0·x[0]', dy=0.55)
        entrada(m0, 'x[0]', '', dy=-0.35, dx=-1.0)
        entrada(m1, f'w0[{j},1]', f'grad_s{j}1·x[1]', dy=0.55)
        entrada(m1, 'x[1]', '', dy=-0.35, dx=-1.0)
        seta(ax, m0, sm, R, R, f's{j}0', f'grad_s{j}0 = grad_s{j}2')
        seta(ax, m1, sm, R, R, f's{j}1', f'grad_s{j}1 = grad_s{j}2')
        seta(ax, sm, bs, R, R, f's{j}2', f'grad_v{j}')
        entrada(bs, f'b0[{j}]', f'grad_v{j}', dx=-0.45, dy=1.0)
        seta(ax, bs, at, R, RB, f'v{j}', f'grad_y{j}·(1−y{j}²)')
        rp = (X_RAMO1, yc)
        ax.add_patch(Circle(rp, 0.07, color=LINHA, zorder=3))
        seta(ax, at, rp, RB, 0.07, f'y{j}', '')
        ramo[j] = rp

    # 2ª camada escondida
    Y2 = {3: 9.4, 4: 4.6}
    X_MUL2, X_SOM2, X_BIAS2, X_ACT2 = 11.4, 13.2, 14.6, 16.6
    for k, yc in Y2.items():
        muls = {j: (X_MUL2, yc + 1.5 * (1 - j)) for j in range(3)}
        sm, bs, at = (X_SOM2, yc), (X_BIAS2, yc), (X_ACT2, yc)
        for j, m in muls.items():
            op(m, '×')
            reta = abs(ramo[j][1] - m[1]) < 1.6
            seta(ax, ramo[j], m, 0.07, R, f'y{j}',
                 f'grad_s{k}{j}·w1[{k - 3},{j}]', t=0.5 if reta else 0.2)
            entrada(m, f'w1[{k - 3},{j}]', f'grad_s{k}{j}·y{j}',
                    dx=-0.12, dy=0.85)
            seta(ax, m, sm, R, R, f's{k}{j}', f'grad_s{k}3')
        op(sm, '+'); op(bs, '+')
        caixa(at, 'tanh', AZUL)
        seta(ax, sm, bs, R, R, f's{k}3', f'grad_v{k}')
        entrada(bs, f'b1[{k - 3}]', f'grad_v{k}', dx=-0.45, dy=1.0)
        seta(ax, bs, at, R, RB, f'v{k}', f'grad_y{k}·(1−y{k}²)')

    # saída
    X_MUL3, X_SOM3, X_BIAS3, X_ACT3, X_ERR, X_L = 18.9, 20.5, 21.9, 24.0, 25.7, 27.2
    YS = 7.0
    mo = {3: (X_MUL3, 8.4), 4: (X_MUL3, 5.6)}
    sm, bs, at, er, lo = ((X_SOM3, YS), (X_BIAS3, YS), (X_ACT3, YS),
                          (X_ERR, YS), (X_L, YS))
    for k, m in mo.items():
        op(m, '×')
        seta(ax, (X_ACT2, Y2[k]), m, RB, R, f'y{k}', f'grad_s5{k - 3}·w2[{k - 3}]')
        entrada(m, f'w2[{k - 3}]', f'grad_s5{k - 3}·y{k}', dx=-0.12, dy=0.85)
        seta(ax, m, sm, R, R, f's5{k - 3}', 'grad_s52')
    op(sm, '+'); op(bs, '+'); op(er, '+')
    caixa(at, 'sigmoide', LARANJA)
    ax.add_patch(Circle(lo, 0.42, facecolor=FUNDO, edgecolor=AZUL, lw=1.4,
                        zorder=3))
    ax.text(*lo, '½(·)²', ha='center', va='center', fontsize=11, zorder=4)
    seta(ax, sm, bs, R, R, 's52', 'grad_v5')
    entrada(bs, 'b2[0]', 'grad_v5', dx=-0.45, dy=1.0)
    seta(ax, bs, at, R, RB, 'v5', 'grad_y5·y5·(1−y5)')
    seta(ax, at, er, RB, R, 'y5', 'grad_e')
    entrada(er, '−d', '', dx=-0.45, dy=1.0)
    seta(ax, er, lo, R, 0.42, 'e', 'grad_L·e')
    seta(ax, (lo[0] + 0.42, lo[1]), (lo[0] + 1.3, lo[1]), 0, 0, 'L',
         'grad_L = 1')

    # títulos e legenda
    for x, t in ((4.1, '1ª camada escondida (tanh)'),
                 (14.0, '2ª camada escondida (tanh)'),
                 (23.3, 'saída (sigmoide) e perda')):
        ax.text(x, 13.4, t, ha='center', fontsize=13, weight='bold',
                color=TEXTO)
    ax.text(0.0, 0.6, 'cinza: variável calculada no forward     ', fontsize=11,
            color=TEXTO_2)
    ax.text(0.0, 0.1, 'vermelho: gradiente de L em relação a ela (backward)',
            fontsize=11, color=GRAD)
    ax.text(8.6, 0.35,
            '• ramificação: cada y da 1ª camada vai para os DOIS neurônios da '
            '2ª, então grad_y0 = grad_s30·w1[0,0] + grad_s40·w1[1,0] '
            '(idem y1, y2)', fontsize=11, color=TEXTO_2)

    ax.set_xlim(-0.3, 28.8)
    ax.set_ylim(-0.2, 13.9)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(PASTA, 'grafo_computacional.png'), dpi=120,
                facecolor=FUNDO)
    plt.close(fig)


if __name__ == '__main__':
    os.makedirs(PASTA, exist_ok=True)
    arquitetura()
    grafo()
    print('figuras salvas em', PASTA)
