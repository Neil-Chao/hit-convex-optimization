import numpy as np
import copy

class Newton:
    def __init__(self, c, epsilon_1=0.001, epsilon_2=0.002, lam=np.array([1, 1])) -> None:
        self.c = c
        self.epsilon_1=epsilon_1
        self.epsilon_2=epsilon_2
        self.lam = lam
        self.d = np.concatenate(tuple((np.expand_dims(np.zeros_like(c), axis=0) for _ in range(len(c.shape)))), axis=0)
        self.dd = np.concatenate(tuple((np.expand_dims(np.zeros_like(c), axis=0) for _ in range(len(c.shape)))), axis=0)
        self.derivative(self.c, self.d, [])
        self.second_derivative(self.d, self.dd)

    def search(self, x, di):
        while True:
            x_next = x + self.lam * di
            x_d = di * self.get_d(x_next, second=False)
            if np.all(np.abs(x_d) < self.epsilon_1):
                return self.lam
            x_dd = di * di * self.get_d(x_next, second=True)
            if np.all(x_dd <= 0):
                raise ValueError("The second derivative is smaller than 0.")
            lam_next = self.lam - x_d / x_dd
            if np.all(np.abs(lam_next - self.lam) < self.epsilon_2):
                return self.lam
            self.lam = lam_next
            # x = x_next

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
            self._second_derivative(x_d, dd, i, [])

    def _second_derivative(self, input, dd, index, index_list: list):
        if len(input.shape) == 1:
            for i in range(input.shape[0]):
                c_list = copy.deepcopy(index_list)
                c_list.append(i)
                d_list = copy.deepcopy(c_list)
                if d_list[index] != 0:
                    d_list[index] -= 1
                    dd[(index,) + tuple(d_list)] = self.d[(index,) + tuple(c_list)] * c_list[index]
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
        res = np.zeros(self.d.shape[0])
        if second:
            for i in range(self.dd.shape[0]):
                res[i] = self._get_d(x, [], self.dd[i])
        else:
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
    # c = np.array([[1, 0, 100],
    #               [-2, 0, 0],
    #               [1, -200, 0],
    #               [0, 0, 0],
    #               [100, 0 ,0]])
    # epsilon_1 = 0.001
    # epsilon_2 = 0.001
    # lam = 0
    # g = Newton(c, epsilon_1, epsilon_2, lam)
    # x = np.array([-1, 1])
    # di = np.array([1, 1])
    # lam = g.search(x, di)
    # print(lam)
    c = np.array([[0, -1, 1],
                  [1, 2, 0],
                  [2, 0, 0]])
    epsilon_1 = 0.001
    epsilon_2 = 0.01
    lam = 0
    n = Newton(c, epsilon_1, epsilon_2, lam)
    x = np.array([0, 0])
    di = np.array([1, 1])
    lam = n.search(x, di)
    print(lam)