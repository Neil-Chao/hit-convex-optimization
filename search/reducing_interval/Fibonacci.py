import numpy as np
import copy

class Fibonacci:
    def __init__(self, c, n, alpha, beta, delta) -> None:
        self.alpha = alpha
        self.beta = beta
        self.c = c
        self.n = n
        self.lam = []
        self.delta = delta
        self.F = self.generate_F()

    def generate_F(self):
        res = np.zeros(self.n+1)
        res[0] = 1
        res[1] = 1
        for i in range(2, self.n+1):
            res[i] = res[i - 1] + res[i - 2]
        return res

    def search(self):
        k = 1
        while True:
            if k == self.n or np.all(self.beta - self.alpha < self.delta):
                return (self.beta + self.alpha) / 2
            lam_k = self.alpha + self.F[self.n-k-1] / self.F[self.n-k+1] * (self.beta - self.alpha)
            mu_k = self.alpha + self.F[self.n-k] / self.F[self.n-k+1] * (self.beta - self.alpha)
            phi_lam = self.output(lam_k)
            phi_mu = self.output(mu_k)
            minus_intervalue = phi_lam - phi_mu
            if minus_intervalue == 0:
                self.alpha = lam_k
                self.beta = mu_k
            elif minus_intervalue < 0:
                self.beta = mu_k
            else:
                self.alpha = lam_k
            k += 1
    
    def output(self, x, index_list: list = []):
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

if __name__ == "__main__":
    alpha = np.array([-1])
    beta = np.array([1])
    c = np.array([-1, -1, 2])
    delta = 0.06
    n = 16
    f = Fibonacci(c, n, alpha, beta, delta)
    index = f.search()
    output = f.output(index)
    print("最小值点为：{}，函数值为：{}".format(index, output))
    alpha = np.array([0])
    beta = np.array([25])
    c = np.array([-1, -21.6, 3])
    f = Fibonacci(c, n, alpha, beta, delta)
    index = f.search()
    output = f.output(index)
    print("最小值点为：{}，函数值为：{}".format(index, output))