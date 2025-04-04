import numpy as np

class Function:
    @staticmethod
    def beale_gradient(x):
        x1, x2 = x
        grad_x1 = 2 * (1.5 - x1 + x1 * x2) * (-1 + x2) + 2 * (2.25 - x1 + x1 * x2**2) * (-1 + x2**2) + 2 * (2.625 - x1 + x1 * x2**3) * (-1 + x2**3)
        grad_x2 = 2 * (1.5 - x1 + x1 * x2) * x1 + 4 * (2.25 - x1 + x1 * x2**2) * x1 * x2 + 6 * (2.625 - x1 + x1 * x2**3) * x1 * x2**2
        return np.array([grad_x1, grad_x2])

    @staticmethod
    def beale_value(x, y):
        return (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2
    
    @staticmethod
    def saddle_gradient(x):
        x1, x2 = x
        grad_x1 = 2 * x1
        grad_x2 = -2 * x2
        return np.array([grad_x1, grad_x2])
    
    @staticmethod
    def saddle_value(x,y):
        return x**2 - y**2

    @staticmethod
    def sphere_gradient(x):
        return 2 * np.array(x)

    @staticmethod
    def sphere_value(x, y):
        return x**2 + y**2

    @staticmethod
    def rosenbrock_gradient(x):
        x1, x2 = x
        grad_x1 = -2 * (1 - x1) - 400 * x1 * (x2 - x1**2)
        grad_x2 = 200 * (x2 - x1**2)
        return np.array([grad_x1, grad_x2])

    @staticmethod
    def rosenbrock_value(x, y):
        return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2 + 1