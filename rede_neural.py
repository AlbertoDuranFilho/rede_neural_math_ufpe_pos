from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras import initializers

X, Y = datasets.make_moons(100, noise=0.1)

color = ['blue' if k == 0 else 'red' for k in Y]

plt.scatter(X[:, 0], X[:, 1], c=color)
plt.savefig('duas_luas.svg')


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def run_neural_net(x, w0, b0, w1, b1, w2, b2):
    # 1ª camada escondida
    s00 = w0[0, 0] * x[0]
    s01 = w0[0, 1] * x[1]
    s02 = s00 + s01
    v0 = s02 + b0[0]
    y0 = tanh(v0)

    s10 = w0[1, 0] * x[0]
    s11 = w0[1, 1] * x[1]
    s12 = s10 + s11
    v1 = s12 + b0[1]
    y1 = tanh(v1)

    s20 = w0[2, 0] * x[0]
    s21 = w0[2, 1] * x[1]
    s22 = s20 + s21
    v2 = s22 + b0[2]
    y2 = tanh(v2)

    # 2ª camada escondida
    s30 = w1[0, 0] * y0
    s31 = w1[0, 1] * y1
    s32 = w1[0, 2] * y2
    s33 = s30 + s31 + s32
    v3 = s33 + b1[0]
    y3 = tanh(v3)

    s40 = w1[1, 0] * y0
    s41 = w1[1, 1] * y1
    s42 = w1[1, 2] * y2
    s43 = s40 + s41 + s42
    v4 = s43 + b1[1]
    y4 = tanh(v4)

    # saída
    s50 = w2[0] * y3
    s51 = w2[1] * y4
    s52 = s50 + s51
    v5 = s52 + b2[0]
    y5 = sigmoid(v5)
    return 1 if y5 > 0.5 else 0


def neural_net(x, d, w0, b0, w1, b1, w2, b2):
    # forward

    # 1ª camada escondida
    s00 = w0[0, 0] * x[0]
    s01 = w0[0, 1] * x[1]
    s02 = s00 + s01
    v0 = s02 + b0[0]
    y0 = tanh(v0)

    s10 = w0[1, 0] * x[0]
    s11 = w0[1, 1] * x[1]
    s12 = s10 + s11
    v1 = s12 + b0[1]
    y1 = tanh(v1)

    s20 = w0[2, 0] * x[0]
    s21 = w0[2, 1] * x[1]
    s22 = s20 + s21
    v2 = s22 + b0[2]
    y2 = tanh(v2)

    # 2ª camada escondida
    s30 = w1[0, 0] * y0
    s31 = w1[0, 1] * y1
    s32 = w1[0, 2] * y2
    s33 = s30 + s31 + s32
    v3 = s33 + b1[0]
    y3 = tanh(v3)

    s40 = w1[1, 0] * y0
    s41 = w1[1, 1] * y1
    s42 = w1[1, 2] * y2
    s43 = s40 + s41 + s42
    v4 = s43 + b1[1]
    y4 = tanh(v4)

    # saída
    s50 = w2[0] * y3
    s51 = w2[1] * y4
    s52 = s50 + s51
    v5 = s52 + b2[0]
    y5 = sigmoid(v5)
    e = y5 - d
    L = 1/2 * (e ** 2)


    # backward
    grad_w0 = np.zeros(w0.shape)
    grad_w1 = np.zeros(w1.shape)
    grad_w2 = np.zeros(w2.shape)
    grad_b0 = np.zeros(b0.shape)
    grad_b1 = np.zeros(b1.shape)
    grad_b2 = np.zeros(b2.shape)

    grad_L = 1
    grad_e = grad_L * e

    # saída
    grad_y5 = grad_e
    grad_v5 = grad_y5 * y5 * (1 - y5)
    grad_b2[0] = grad_v5
    grad_s52 = grad_v5
    grad_s50 = grad_s52
    grad_s51 = grad_s52
    grad_w2[0] = grad_s50 * y3
    grad_y3 = grad_s50 * w2[0]
    grad_w2[1] = grad_s51 * y4
    grad_y4 = grad_s51 * w2[1]

    # 2ª camada escondida
    grad_v3 = grad_y3 * (1 - y3 ** 2)
    grad_v4 = grad_y4 * (1 - y4 ** 2)

    grad_b1[0] = grad_v3
    grad_b1[1] = grad_v4
    grad_s33 = grad_v3
    grad_s43 = grad_v4

    grad_s30 = grad_s33
    grad_s31 = grad_s33
    grad_s32 = grad_s33
    grad_s40 = grad_s43
    grad_s41 = grad_s43
    grad_s42 = grad_s43

    grad_w1[0, 0] = grad_s30 * y0
    grad_w1[0, 1] = grad_s31 * y1
    grad_w1[0, 2] = grad_s32 * y2
    grad_w1[1, 0] = grad_s40 * y0
    grad_w1[1, 1] = grad_s41 * y1
    grad_w1[1, 2] = grad_s42 * y2

    grad_y0 = grad_s30 * w1[0, 0] + grad_s40 * w1[1, 0]
    grad_y1 = grad_s31 * w1[0, 1] + grad_s41 * w1[1, 1]
    grad_y2 = grad_s32 * w1[0, 2] + grad_s42 * w1[1, 2]

    grad_v0 = grad_y0 * (1 - y0 ** 2)
    grad_v1 = grad_y1 * (1 - y1 ** 2)
    grad_v2 = grad_y2 * (1 - y2 ** 2)

    grad_b0[0] = grad_v0
    grad_b0[1] = grad_v1
    grad_b0[2] = grad_v2
    grad_s02 = grad_v0
    grad_s12 = grad_v1
    grad_s22 = grad_v2

    grad_s00 = grad_s02
    grad_s01 = grad_s02
    grad_s10 = grad_s12
    grad_s11 = grad_s12
    grad_s20 = grad_s22
    grad_s21 = grad_s22
    grad_w0[0, 0] = grad_s00 * x[0]
    grad_w0[0, 1] = grad_s01 * x[1]
    grad_w0[1, 0] = grad_s10 * x[0]
    grad_w0[1, 1] = grad_s11 * x[1]
    grad_w0[2, 0] = grad_s20 * x[0]
    grad_w0[2, 1] = grad_s21 * x[1]
    return grad_w0, grad_b0, grad_w1, grad_b1, grad_w2, grad_b2, L

def main():
    # inicialização aleatória
    w0 = np.random.rand(3, 2)
    w1 = np.random.rand(2, 3)
    w2 = np.random.rand(2)
    b0 = np.random.rand(3)
    b1 = np.random.rand(2)
    b2 = np.random.rand(1)

    iniciais = [w0.copy(), b0.copy(), w1.copy(), b1.copy(), w2.copy(), b2.copy()]

    taxa = 0.1

    acc = 0
    for i in range(100):
        out = run_neural_net(X[i], w0, b0, w1, b1, w2, b2)
        if out == Y[i]:
            acc += 1
    print(acc, "acurácia antes do treinamento")

    # gradiente descendente
    for i in range(10000):
        loss = 0

        grad_w0 = np.zeros(w0.shape)
        grad_w1 = np.zeros(w1.shape)
        grad_w2 = np.zeros(w2.shape)
        grad_b0 = np.zeros(b0.shape)
        grad_b1 = np.zeros(b1.shape)
        grad_b2 = np.zeros(b2.shape)

        for k in range(100):
            g_w0, g_b0, g_w1, g_b1, g_w2, g_b2, L = neural_net(X[k], Y[k], w0, b0, w1, b1, w2, b2)

            grad_w0 += g_w0
            grad_w1 += g_w1
            grad_w2 += g_w2
            grad_b0 += g_b0
            grad_b1 += g_b1
            grad_b2 += g_b2
            loss += L

        w0 -= taxa * grad_w0
        w1 -= taxa * grad_w1
        w2 -= taxa * grad_w2
        b0 -= taxa * grad_b0
        b1 -= taxa * grad_b1
        b2 -= taxa * grad_b2

        if i % 1000 == 0:
            print(i, loss)

    acc = 0
    for i in range(100):
        out = run_neural_net(X[i], w0, b0, w1, b1, w2, b2)
        if out == Y[i]:
            acc += 1
    print('acc', acc)

    keras.config.set_floatx('float64')
    model = Sequential()
    model.add(Dense(3, input_dim=2, activation='tanh'))
    model.add(Dense(2, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))

    w0_i, b0_i, w1_i, b1_i, w2_i, b2_i = iniciais
    model.set_weights([w0_i.T, b0_i, w1_i.T, b1_i, w2_i.reshape(2, 1), b2_i])

    taxa_keras = taxa * len(X) / 2
    opt = SGD(learning_rate=taxa_keras)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])

    model.fit(np.tile(X, (10000, 1)), np.tile(Y, 10000), epochs=1,
              batch_size=len(X), shuffle=False, verbose=False)

    loss_keras, acc_keras = model.evaluate(X, Y, verbose=False)
    print('acc keras', round(acc_keras * 100))

    # compara os pesos finais das duas redes
    finais = [w0.T, b0, w1.T, b1, w2.reshape(2, 1), b2]
    diferenca = max(np.max(np.abs(p - k)) for p, k in zip(finais, model.get_weights()))
    print('maior diferença entre os pesos finais (manual x keras):', diferenca)


main()
