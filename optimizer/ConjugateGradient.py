import sys
sys.path.append(r"D:\code\python_src\python-demo\07-optimization")

from search.interpolation_newton.Newton import Newton
import numpy as np

class ConjugateGradient(Newton):
    def __init__(self, c, x, epsilon=1e-2, epsilon_1=0.001, epsilon_2=0.002, lam=np.array([1, 1])) -> None:
        super(ConjugateGradient, self).__init__(c, epsilon_1, epsilon_2, lam)
        self.x = x
        self.epsilon=epsilon
        self.di = np.zeros((1, x.shape[0]))

    def solve(self):
        k = 0
        self.di[0] = -self.get_d(self.x)
        while True:
            this_d = self.get_d(self.x)
            if np.linalg.norm(this_d) < self.epsilon:
                return self.x
            lam = self.search(self.x, self.di[k])
            next_x = self.x + lam * self.di[k]
            next_d = self.get_d(next_x)
            beta_k = np.dot(next_d, next_d) / np.dot(this_d, this_d)
            next_di = -next_d + beta_k * self.di[k]
            self.di = np.concatenate((self.di, np.expand_dims(next_di, axis=0)))
            self.x = next_x
            self.lam = np.array([1, 1])
            k += 1

if __name__ == "__main__":
    # c = np.array([[0, -1, 1],
    #               [1, 2, 0],
    #               [2, 0, 0]])
    # x = np.array([0, 0])
    # cg = ConjugateGradient(c, x)
    # x = cg.solve()
    # print(x)

    c = np.array([[60, -4, 1],
                  [-10, -1, 0],
                  [1, 0, 0]])
    x = np.array([0, 0])
    cg = ConjugateGradient(c, x)
    x = cg.solve()
    output = cg.output(x)
    print("最小值点为：{}，函数值为：{}".format(x, output))


