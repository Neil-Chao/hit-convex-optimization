import numpy as np
import copy

class GoldsteinPrice:
    def __init__(self, c, rho=0.1, alpha=1.5, beta=0.5, lam=1) -> None:
        self.c = c
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.lam = lam
        self.d = np.concatenate(tuple((np.expand_dims(np.zeros_like(c), axis=0) for _ in range(len(c.shape)))), axis=0)
        self.dd = np.concatenate(tuple(np.expand_dims(np.concatenate(tuple((np.expand_dims(np.zeros_like(c), axis=0) for _ in range(len(c.shape)))), axis=0), axis=0) for _ in range(len(c.shape)) ))
        self.derivative(self.c, self.d, [])
        self.second_derivative(self.d, self.dd)

    def search(self, x):
        x_d = self.get_d(x)
        x_dd = self.get_d(x, second=True)
        if self.is_positive_definite(x_dd):
            inverse_dd = np.linalg.inv(x_dd)
            di = -(inverse_dd @ x_d)
        else:
            di = -x_d
        phi_1 = self.output(x, [])
        phi_1_d = np.dot(self.get_d(x), di)
        while True:
            phi_2 = self.output(x + self.lam * di, [])
            if np.all(phi_2 <= (phi_1 + self.rho * phi_1_d * self.lam)):
                if np.all(phi_2 >= (phi_1 + (1 - self.rho) * phi_1_d * self.lam)):
                    return self.lam
                else:
                    self.lam *= self.alpha
            else:
                self.lam *= self.beta

    def is_positive_definite(self, m):
        return np.all(np.linalg.eigvals(m) > 0)

    def derivative(self, input, m, index_list: list):
        if len(input.shape) == 1:
            for i in range(input.shape[0]):
                c_list = copy.deepcopy(index_list)
                c_list.append(i)
                for j in range(len(self.c.shape)):
                    d_list = copy.deepcopy(c_list)
                    if d_list[j] != 0:
                        d_list[j] -= 1
                        m[(j,) + tuple(d_list)] = self.c[tuple(c_list)] * c_list[j]
            return
        for i in range(input.shape[0]):
            row = input[i]
            self.derivative(row, m, index_list + [i])

    def second_derivative(self, d, dd):
        for i in range(d.shape[0]):
            x_d = d[i]
            for j in range(d.shape[0]):
                self._second_derivative(x_d, dd, [i, j], [])

    def _second_derivative(self, input, dd, index, index_list: list):
        if len(input.shape) == 1:
            i = index[0]
            j = index[1]
            for k in range(input.shape[0]):
                c_list = copy.deepcopy(index_list)
                c_list.append(k)
                d_list = copy.deepcopy(c_list)
                if d_list[j] != 0:
                    d_list[j] -= 1
                    dd[(i,j,) + tuple(d_list)] = self.d[(i,) + tuple(c_list)] * c_list[j]


            return
        for i in range(input.shape[0]):
            row = input[i]
            self._second_derivative(row, dd, index, index_list + [i])

    def output(self, x, index_list: list=[]):
        res = 0
        row = self.c[tuple(index_list)]
        if len(row.shape) == 1:
            for i in range(row.shape[0]):
                c_list = copy.deepcopy(index_list)
                c_list.append(i)
                tmp = row[i]
                for j in range(len(c_list)):
                    tmp *= np.power(x[j], c_list[j])
                res += tmp
            return res
        for i in range(row.shape[0]):
            res += self.output(x, index_list + [i])
        return res

    def get_d(self, x, second=False):
        if second:
            res = np.zeros((self.d.shape[0], self.d.shape[0]))
            for i in range(self.dd.shape[0]):
                for j in range(self.dd.shape[0]):
                    res[i][j] = self._get_d(x, [], self.dd[i][j])
        else:
            res = np.zeros(self.d.shape[0])
            for i in range(self.d.shape[0]):
                res[i] = self._get_d(x, [], self.d[i])
        return res
        

    def _get_d(self, x, index_list, d):
        res = 0
        row = d[tuple(index_list)]
        if len(row.shape) == 1:
            for i in range(row.shape[0]):
                d_list = copy.deepcopy(index_list)
                d_list.append(i)
                tmp = row[i]
                for j in range(len(d_list)):
                    tmp *= np.power(x[j], d_list[j])
                res += tmp
            return res
        for i in range(row.shape[0]):
            res += self._get_d(x, index_list + [i], d)
        return res


if __name__ == "__main__":
    c = np.array([[1, 0, 100],
                  [-2, 0, 0],
                  [1, -200, 0],
                  [0, 0, 0],
                  [100, 0 ,0]])
    rho = 0.1
    alpha = 1.5
    beta = 0.5
    gp = GoldsteinPrice(c, rho, alpha, beta)
    x = np.array([-1, 1])
    lam = gp.search(x)
    print(lam)
    print(x + lam)
    # c = np.array([[0, -1, 1],
    #               [1, 2, 0],
    #               [2, 0, 0]])
    # epsilon_1 = 0.001
    # epsilon_2 = 0.01
    # lam = 0
    # g = GoldsteinPrice(c, epsilon_1, epsilon_2, lam)
    # x = np.array([0, 0])
    # lam = g.search(x)
    # print(lam)