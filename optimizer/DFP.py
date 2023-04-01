import sys
sys.path.append(r"D:\code\python_src\python-demo\07-optimization")

from search.interpolation_newton.Newton import Newton
import numpy as np

class DFP(Newton):
    def __init__(self, c, x, H, epsilon=1e-2, epsilon_1=0.001, epsilon_2=0.002, lam=np.array([1, 1])) -> None:
        super(DFP, self).__init__(c, epsilon_1, epsilon_2, lam)
        self.x = x
        self.H = H
        self.len = self.x.shape[0]
        self.epsilon=epsilon
        self.di = np.zeros((0, x.shape[0]))

    def solve(self):
        k = 0
        while True:
            this_d = self.get_d(self.x)
            if np.linalg.norm(this_d) < self.epsilon:
                return self.x
            di = -self.H @ this_d.reshape(self.len)
            self.di = np.concatenate((self.di, np.expand_dims(di, axis=0)))
            lam = self.search(self.x, self.di[k])
            next_x = self.x + lam * self.di[k]
            next_d = self.get_d(next_x)
            s = (lam * self.di[k]).reshape(self.len, 1)
            y = (next_d - this_d).reshape(self.len, 1)
            H_next = self.H + (s @ s.T) / (s.T @ y) - (self.H @ y @ y.T @ self.H) / (y.T @ self.H @ y)
            self.H = H_next
            self.x = next_x
            k += 1

if __name__ == "__main__":
    # c = np.array([[0, 0, 1],
    #               [0, 0, 0],
    #               [10, 0, 0]])
    # x = np.array([0.1, 1])
    # H = np.array([[1, 0], [0, 1]])
    # cg = DFP(c, x, H)
    # x = cg.solve()
    # print(x)

    c = np.array([[60, -4, 1],
                  [-10, -1, 0],
                  [1, 0, 0]])
    x = np.array([0, 0])
    H = np.array([[1, 0], [0, 1]])
    dfp = DFP(c, x, H)
    x = dfp.solve()
    output = dfp.output(x)
    print("最小值点为：{}，函数值为：{}".format(x, output))