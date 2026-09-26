"""
Confere o backward escrito à mão em rede_neural.py com a definição de
derivada:  dL/dw ~ (L(w + h) - L(w - h)) / 2h

Carrega as funções de rede_neural.py sem executar o treinamento (tudo a partir
de "def main" é descartado) e compara, para cada um dos 20 parâmetros, o
gradiente de neural_net() com a diferença finita, em 20 exemplos aleatórios.

Uso:  python verifica_gradiente.py
"""
import numpy as np

codigo = open('rede_neural.py', encoding='utf-8').read()
codigo = codigo[:codigo.index('def main')]
codigo = codigo.replace("plt.savefig('duas_luas.svg')", '')
ns = {}
exec(codigo, ns)
neural_net = ns['neural_net']

nomes = ['w0', 'b0', 'w1', 'b1', 'w2', 'b2']
formatos = [(3, 2), (3,), (2, 3), (2,), (2,), (1,)]
h = 1e-6
rng = np.random.default_rng(0)
pior = {n: 0.0 for n in nomes}

for teste in range(20):
    params = [rng.normal(size=f) for f in formatos]
    x = rng.normal(size=2)
    d = teste % 2
    grads = neural_net(x, d, *params)[:6]
    for nome, p, g in zip(nomes, params, grads):
        for ix in np.ndindex(p.shape):
            original = p[ix]
            p[ix] = original + h
            mais = neural_net(x, d, *params)[-1]
            p[ix] = original - h
            menos = neural_net(x, d, *params)[-1]
            p[ix] = original
            numerico = (mais - menos) / (2 * h)
            erro = abs(numerico - g[ix]) / max(1e-8, abs(numerico) + abs(g[ix]))
            pior[nome] = max(pior[nome], erro)

print('maior erro relativo (backward x diferença finita):')
for nome in nomes:
    print(f'  {nome}: {pior[nome]:.1e}')
