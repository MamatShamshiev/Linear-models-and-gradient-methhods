import numpy as np
import scipy.special as sc
import scipy


class BinaryLogistic():
    """
    Оракул для задачи двухклассовой логистической регрессии.

    Оракул должен поддерживать l2 регуляризацию.
    """

    def __init__(self, l2_coef):
        """
        Задание параметров оракула.

        l2_coef - коэффициент l2 регуляризации
        """

        self.lambda_2 = l2_coef

    def func(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w - одномерный numpy array
        """
        return ((1 / X.shape[0]) * (np.logaddexp((-y) * X.dot(w), 0)).sum() +
                0.5 * self.lambda_2 * (np.linalg.norm(w) ** 2))

    def grad(self, X, y, w):
        """
        Вычислить градиент функционала в точке w на выборке X с ответами y.

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w - одномерный numpy array
        """
        return ((-1 / X.shape[0]) * X.T.dot(y * sc.expit(-y * X.dot(w))) +
                self.lambda_2 * w)

    
class MulticlassLogistic():
    """
    Оракул для задачи многоклассовой логистической регрессии.

    Оракул должен поддерживать l2 регуляризацию.

    w в этом случае двумерный numpy array размера (class_number, d),
    где class_number - количество классов в задаче, d - размерность задачи
    """

    def __init__(self, l2_coef, class_number=None):
        """
        Задание параметров оракула.

        class_number - количество классов в задаче

        l2_coef - коэффициент l2 регуляризации
        """
        self.class_number = class_number
        self.lambda_2 = l2_coef

    def func(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array, классы от 0 до class_number-1

        w - двумерный numpy array
        """
        M = X.dot(w.T)
        return (((-1 / X.shape[0]) * 
                (M[np.arange(X.shape[0]), y].sum() - (sc.logsumexp(M, axis=1)).sum())) +
                0.5 * self.lambda_2 * (np.linalg.norm(w) ** 2))
                
    def grad(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array, классы от 0 до class_number-1

        w - двумерный numpy array
        """
        if self.class_number is None:
            self.class_number = np.amax(y) + 1
        M = X.dot(w.T)
        M_max = M.max(axis=1)[:, np.newaxis]
        Exp_matrix = np.exp(M - M_max) / np.exp(sc.logsumexp(M - M_max, axis=1))[:, np.newaxis]
        Y = y[:, np.newaxis] == np.mgrid[0:len(y), 0:self.class_number][1]
        return (-1 / X.shape[0]) * (X.T.dot(Y).T - X.T.dot(Exp_matrix).T) + self.lambda_2 * w
