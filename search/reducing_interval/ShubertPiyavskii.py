import numpy as np
import copy

class ShubertPiyavskii:
    def __init__(self, c, alpha, beta, delta) -> None:
        self.alpha = alpha
        self.beta = beta
        self.c = c
        self.delta = delta
        self.d = np.concatenate(tuple((np.expand_dims(np.zeros_like(c), axis=0) for _ in range(len(c.shape)))), axis=0)
        self.derivative(self.c, [])

    def max_d(self):
        alpha_d = np.abs(self.get_d(self.alpha))
        beta_d = np.abs(self.get_d(self.beta))
        if np.all(alpha_d > beta_d):
            return alpha_d
        return beta_d

    def search(self):
        x = (self.alpha + self.beta) / 2
        alpha = copy.deepcopy(self.alpha)
        beta = copy.deepcopy(self.beta)
        upper_sawtooth = [(alpha, self.output(alpha)), (beta, self.output(beta))]
        lower_sawtooth = []
        k = 1
        max_d = self.max_d()
        while True:
            if k != 1:
                y, x = self.find_lowest(lower_sawtooth)
            phi_x = self.output(x)
            if k != 1:
                x_gap = phi_x - y
                if x_gap < self.delta:
                    return x
            alpha, beta = self.find_neighbors(copy.deepcopy(upper_sawtooth), x)
            phi_alpha = self.output(alpha)
            phi_beta = self.output(beta)
            
            lam_k = self.intersection(phi_alpha, phi_x, alpha, x,  max_d)
            mu_k = self.intersection(phi_x, phi_beta, x, beta, max_d)
            # phi_lam = self.output(lam_k)
            # phi_mu = self.output(mu_k)
            lower_sawtooth.append((lam_k, phi_x + max_d * (lam_k - x)))
            lower_sawtooth.append((mu_k, phi_x - max_d * (mu_k - x)))
            upper_sawtooth.append((x, phi_x))
            k += 1


            
    
    def find_lowest(self, lower_sawtooth: list):
        index = None
        min = np.inf
        min_x = None
        for i in range(len(lower_sawtooth)):
            x, y = lower_sawtooth[i]
            if y < min:
                min = y
                index = i
                min_x = x
        lower_sawtooth.pop(index)
        return (min, min_x)
    
    def find_neighbors(self, upper_sawtooth: list, x):
        upper_sawtooth.sort(key=lambda x: x[0])
        for i in range(len(upper_sawtooth)):
            if np.all(upper_sawtooth[i][0] > x):
                return (upper_sawtooth[i-1][0], upper_sawtooth[i][0])
                
        

    def intersection(self, y_0, y_1, x_0, x_1, d):
        return (y_0 - y_1) / (2 * d) + (x_0 + x_1) / 2

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


if __name__ == "__main__":
    alpha = np.array([-1])
    beta = np.array([1])
    c = np.array([-1, -1, 2])
    delta = 0.0001
    sp = ShubertPiyavskii(c, alpha, beta, delta)
    index = sp.search()
    output = sp.output(index)
    print(index, output)