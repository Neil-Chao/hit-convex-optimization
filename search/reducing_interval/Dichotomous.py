import numpy as np

class Dichotomous:
    def __init__(self, c, b, alpha, beta, delta) -> None:
        self.alpha = alpha
        self.beta = beta
        self.k = 0
        self.c = c
        self.b = b
        self.lam = []
        self.delta = delta

    def search(self):
        while True:
            middle = (self.beta[:, self.k] + self.alpha[:, self.k]) / 2
            middle_d = self.derivative(middle)
            if np.all(np.abs(middle_d) < self.delta):
                return middle
            elif np.all(middle_d > 0):
                self.alpha = np.concatenate((self.alpha, np.expand_dims(self.alpha[:, -1], axis=0)), axis=1)
                self.beta = np.concatenate((self.beta, np.expand_dims(middle, axis=0)), axis=1)
            else:
                self.alpha = np.concatenate((self.alpha, np.expand_dims(middle, axis=0)), axis=1)
                self.beta = np.concatenate((self.beta, np.expand_dims(self.beta[:, -1], axis=0)), axis=1)
            self.k += 1

    def derivative(self, input):
        assert input.shape[0] == self.c.shape[0]
        d = np.zeros((self.c.shape[0], self.c.shape[1]))
        res = np.zeros(self.c.shape[0])
        for i in range(self.c.shape[0]):
            index_c = self.c[i]
            for j in range(index_c.shape[0]):
                if j==0:
                    d[i][j] = index_c[j]
                else:
                    d[i][j] = (j + 1) * index_c[j]
            tmp_value = 0
            for j in range(d[i].shape[0]):
                if j == 0:
                    tmp_value += d[i][j]
                else:
                    tmp_value += d[i][j] * np.power(input[i], j)
            res[i] = tmp_value
        return res
    
    def output(self, input):
        assert input.shape[0] == self.c.shape[0]
        res = 0
        for i in range(self.c.shape[0]):
            var_list = self.c[i] 
            for j in range(var_list.shape[0]):
                res += var_list[j] * np.power(input[i], j+1)
        res += self.b
        return res

if __name__ == "__main__":
    alpha = np.array([[-1]])
    beta = np.array([[1]])
    c = np.array([[-1, 2]])
    b = -1
    delta = 0.006
    dich = Dichotomous(c, b, alpha, beta, delta)
    index = dich.search()
    output = dich.output(index)
    print("最小值点为：{}，函数值为：{}".format(index, output))

    alpha = np.array([[0]])
    beta = np.array([[25]])
    c = np.array([[-21.6, 3]])
    b = -1
    delta = 0.08
    dich = Dichotomous(c, b, alpha, beta, delta)
    index = dich.search()
    output = dich.output(index)
    print("最小值点为：{}，函数值为：{}".format(index, output))