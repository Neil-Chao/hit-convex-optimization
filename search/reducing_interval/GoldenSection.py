import numpy as np

class GoldenSection:
    def __init__(self, c, b, alpha, beta, delta) -> None:
        self.alpha = alpha
        self.beta = beta
        self.k = 0
        self.gamma = 0.618
        self.c = c
        self.b = b
        self.lam = []
        self.delta = delta

    def search(self):
        flag = True
        while True:
            if flag:
                lam_k = self.alpha[:, self.k] + (1 - self.gamma) * (self.beta[:, self.k] - self.alpha[:, self.k])
                mu_k = self.alpha[:, self.k] + self.gamma * (self.beta[:, self.k] - self.alpha[:, self.k])
                phi_lam = self.output(lam_k)
                phi_mu = self.output(mu_k)
            if np.all(self.beta[:, self.k] - self.alpha[:, self.k] < self.delta):
                return (self.beta[:, self.k] + self.alpha[:, self.k]) / 2
            minus_intervalue = phi_lam - phi_mu
            if minus_intervalue == 0:
                flag = True
                self.alpha = np.concatenate((self.alpha, np.expand_dims(lam_k, axis=0)), axis=1)
                self.beta = np.concatenate((self.beta, np.expand_dims(mu_k, axis=0)), axis=1)
            elif minus_intervalue < 0:
                flag = False
                self.alpha = np.concatenate((self.alpha, np.expand_dims(self.alpha[:, -1], axis=0)), axis=1)
                self.beta = np.concatenate((self.beta, np.expand_dims(mu_k, axis=0)), axis=1)
                mu_k = lam_k
                lam_k = self.alpha[:, self.k + 1] + (1 - self.gamma) * (self.beta[:, self.k + 1] - self.alpha[:, self.k + 1])
                phi_mu = phi_lam
                phi_lam = self.output(lam_k)
            else:
                flag = False
                self.alpha = np.concatenate((self.alpha, np.expand_dims(lam_k, axis=0)), axis=1)
                self.beta = np.concatenate((self.beta, np.expand_dims(self.beta[:, -1], axis=0)), axis=1)
                lam_k = mu_k
                mu_k = self.alpha[:, self.k + 1] + self.gamma * (self.beta[:, self.k + 1] - self.alpha[:, self.k + 1])
                phi_lam = phi_mu
                phi_mu = self.output(mu_k)
            self.k += 1
    
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
    delta = 0.06
    gs = GoldenSection(c, b, alpha, beta, delta)
    index = gs.search()
    output = gs.output(index)
    print(index, output)
    alpha = np.array([[0]])
    beta = np.array([[25]])
    c = np.array([[-21.6, 3]])
    b = -1
    delta = 0.08
    gs = GoldenSection(c, b, alpha, beta, delta)
    index = gs.search()
    output = gs.output(index)
    print(index, output)