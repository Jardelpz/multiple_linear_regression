import math
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as scipy
import numpy as np

from sklearn.linear_model import LinearRegression


class Main:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys
        self.avgX = self.avg(xs)
        self.avgY = self.avg(ys)
        self.beta = None

    @staticmethod
    def avg(items):
        return sum(items) / len(items)

    def corrcoef(self):
        dividend = 0
        sum_quadratic_x = 0
        sum_quadratic_y = 0
        for i, _ in enumerate(self.xs):
            dividend += (self.xs[i] - self.avgX) * (self.ys[i] - self.avgY)
            sum_quadratic_x += math.pow(self.xs[i] - self.avgX, 2)
            sum_quadratic_y += math.pow(self.ys[i] - self.avgY, 2)

        divisor = math.sqrt(sum_quadratic_x * sum_quadratic_y)
        return round(dividend / divisor, 4)

    def regression(self):
        dividend = 0
        sum_quadratic_x = 0
        for i, _ in enumerate(self.xs):
            dividend = dividend + (self.xs[i] - self.avgX) * (self.ys[i] - self.avgY)
            sum_quadratic_x += math.pow(self.xs[i] - self.avgX, 2)

        b1 = round(dividend / sum_quadratic_x, 4)
        b0 = round(self.avgY - (b1 * self.avgX), 4)
        return b0, b1


class MultipleRegression:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.beta = None

    def multiple_linear_regression(self):
        transpose_x = self.transpose(self.X)

        # Calcula a inversa de X^T * X
        XTX = self.multiply(transpose_x, self.X)
        XTX_inv = self.inverse(XTX)

        # Calcula os coeficientes beta
        XTy = self.multiply(transpose_x, self.y)
        beta = self.multiply(XTX_inv, XTy)
        self.beta = beta
        return beta

    def predict(self, x_pred):
        return self.multiply(self.beta, x_pred)

    @staticmethod
    def transpose(matrix):
        return matrix.T
        # return np.array([[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))])

    @staticmethod
    def multiply(a, b):
        return a.dot(b)

    # Função para calcular a inversa de uma matriz
    @staticmethod
    def inverse(matrix):
        return np.linalg.inv(matrix)

    def plot_new_3d(self, X, size, room, price):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        predicts = []
        for x_pred in X:
            predicts.append(self.predict(x_pred))

        ax.scatter(size, room, price, c='r', marker='o')

        # Definindo os rótulos dos eixos
        ax.plot(size, room, predicts, color='green')
        ax.set_xlabel('Tamanho da casa')
        ax.set_ylabel('Número de quartos')
        ax.set_zlabel('Preço da casa')
        plt.show()


mat = scipy.loadmat('data.mat')
data = mat['data']

size = []
rooms = []
price = []

for i in data:
    size.append(i[0])
    rooms.append(i[1])
    price.append(float(i[2]))

prices_d = pd.DataFrame(price)
rooms_d = pd.DataFrame(rooms)
size_d = pd.DataFrame(size)


def find_room(price):
    for i in data:
        if i[2] == price:
            return i[1]


def multiply(a, b):
    return a.dot(b)


def show_graph(xs, ys, origin):
    main = Main(xs=xs, ys=ys)
    coef = main.corrcoef()
    b0, b1 = main.regression()

    plt.scatter(xs, ys)
    plt.title(f'{origin}: Coef {coef}, b0: {b0}, b1: {b1}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.plot(xs, [b1 * i + b0 for i in xs], color='red')
    plt.show()
    return coef, b0, b1


def already_multiple_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model.coef_


coef_room, b0_room, b1_room = show_graph(rooms, price, 'Room')
coef_size, b0_size, b1_size = show_graph(size, price, 'Size')

# criar a matriz X com os valores de tamanho e número de quartos
X = np.array([[1, i[0], i[1]] for i in data])
y = price

mlr = MultipleRegression(X, y)
beta = mlr.multiple_linear_regression()

# test
x_pred = [1, 1650, 3]
y_pred = mlr.predict(x_pred)

print(f'Média do preço das casas: {prices_d.mean()[0]} m2')
print(f'Menor preço das casas: {prices_d.min()[0]}')
print(f'Quartos da casa mais cara: {find_room(prices_d.max()[0])}')
print(f'Comparando preço com modelos prontos: {multiply(already_multiple_regression(X, y), x_pred)}')
print('Modelo implementado, Price:', round(y_pred, 4))
mlr.plot_new_3d(X, size, rooms, price)
