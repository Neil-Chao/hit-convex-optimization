import numpy as np
import copy

class WolfePowell:
    def __init__(self, c, rho=0.1, sigma=0.9, alpha=1.5, beta=0.5, lam=1) -> None:
        self.c = c
        self.rho = rho
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta
        self.lam = lam
        self.d = np.concatenate(tuple((np.expand_dims(np.zeros_like(c), axis=0) for _ in range(len(c.shape)))), axis=0)
        self.derivative(self.c, [])

    def search(self, x, di):
        phi_1 = self.output(x, [])
        phi_1_d = np.dot(self.get_d(x), di)
        while True:
            phi_2 = self.output(x + self.lam * di, [])
            if np.all(phi_2 <= (phi_1 + self.rho * phi_1_d * self.lam)):
                phi_2_d = np.dot(self.get_d(x + self.lam * di), di)
                if np.all(phi_2_d >= (self.sigma * phi_1_d * self.lam)):
                    return self.lam
                else:
                    self.lam *= self.alpha
            else:
                self.lam *= self.beta

    def derivative(self, input, index_list: list):
        if len(input.shape) == 1:
            for i in range(input.shape[0]):
                c_list = copy.deepcopy(index_list)
                c_list.append(i)
                for j in range(len(self.c.shape)):
                    d_list = copy.deepcopy(c_list)
                    if d_list[j] != 0:
                        d_list[j] -= 1
                        self.d[(j,) + tuple(d_list)] = self.c[tuple(c_list)] * c_list[j]
            return
        for i in range(input.shape[0]):
            row = input[i]
            self.derivative(row, index_list + [i])

    def output(self, x, index_list: list):
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

    def get_d(self, x):
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
    wp = WolfePowell(c, rho, alpha, beta)
    x = np.array([-1, 1])
    di = np.array([1, 1])
    lam = wp.search(x, di)
    print(lam)