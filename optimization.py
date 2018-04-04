from oracles import BinaryLogistic, MulticlassLogistic
import time
import numpy as np
import scipy.special as sc
from sklearn.model_selection import train_test_split


class GDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """
    def __init__(self, loss_function, step_alpha=1, step_beta=0, 
                 tolerance=1e-5, max_iter=1000, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора. 
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        - 'multinomial_logistic' - многоклассовая логистическая регрессия
                
        step_alpha - float, параметр выбора шага из текста задания
        
        step_beta- float, параметр выбора шага из текста задания
        
        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию.
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если (f(x_{k+1}) - f(x_{k})) < tolerance: то выход 
        
        max_iter - максимальное число итераций     
        
        **kwargs - аргументы, необходимые для инициализации   
        """
        
        self.loss_function = loss_function
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.kwargs = kwargs
        
    def fit(self, X, y, w_0=None, trace=False, calc_accuracy=False):
        """
        Обучение метода по выборке X с ответами y
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        w_0 - начальное приближение в методе
        
        trace - переменная типа bool
      
        Если trace = True, то метод должен вернуть словарь history, содержащий информацию 
        о поведении метода. Длина словаря history = количество итераций + 1 (начальное приближение)
        
        history['time']: list of floats, содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит значения функции на каждой итерации
        (0 для самой первой точки)
        """
        
        y = y.astype(int)
        if calc_accuracy is True:
            X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                                train_size=0.7, random_state=11)
        else:
            X_train = X
            y_train = y
            
        if trace is True:
            history = {}

        if w_0 is None:
            if self.loss_function == 'binary_logistic':
                self.w = np.random.uniform(-1, 1, X_train.shape[1])
            elif self.loss_function == 'multinomial_logistic':
                self.w = np.random.uniform(-1, 1, (np.amax(y_train) + 1, X_train.shape[1]))
        else:
            self.w = w_0

        if self.loss_function == 'binary_logistic':
            self.oracle = BinaryLogistic(**self.kwargs)
        elif self.loss_function == 'multinomial_logistic':
            self.oracle = MulticlassLogistic(**self.kwargs)

        f = self.get_objective(X, y)
        if trace is True:
            history['func'] = [f]
            history['time'] = [0]
            start = time.clock()
        iteration = 1    
        f_pred = f
        self.w = self.w - self.step_alpha / (iteration ** self.step_beta) * self.get_gradient(X, y)
        f = self.get_objective(X, y)
        if trace is True:
            history['func'].append(f)
            history['time'].append(time.clock() - start)
            start = time.clock()
        iteration += 1
        
        while (iteration <= self.max_iter and abs(f - f_pred) > self.tolerance):
            f_pred = f
            self.w = (self.w - (self.step_alpha / (iteration ** self.step_beta)) * 
                      self.get_gradient(X, y))           
            f = self.get_objective(X, y)
            if trace is True:
                history['func'].append(f)
                history['time'].append(time.clock() - start)
            iteration += 1
        if trace is True:
            return history 
        else:
            return self
                    
    def predict(self, X):
        """
        Получение меток ответов на выборке X
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        return: одномерный numpy array с предсказаниями
        """
        probs = self.predict_proba(X)
        if self.loss_function == 'binary_logistic':
            return np.where(probs[:, 0] > probs[:, 1], -1, 1)
        else:
            return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        """
        Получение вероятностей принадлежности X к классу k
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        return: двумерной numpy array, [i, k] значение соответветствует вероятности
        принадлежности i-го объекта к классу k 
        """
        
        if self.w is None:
            raise Exception('Not trained yet')

        if self.loss_function == 'binary_logistic':    
            probs1 = sc.expit(X.dot(self.w))
            return np.hstack(((1 - probs1)[:, np.newaxis], probs1[:, np.newaxis]))
        else:
            M = X.dot(self.w.T)
            M_max = M.max(axis=1)[:, np.newaxis]
            Exp_matrix = np.exp(sc.logsumexp(M - M_max, axis=1))
            return np.exp(M - M_max) / Exp_matrix[:, np.newaxis]
            
    def get_objective(self, X, y):
        """
        Получение значения целевой функции на выборке X с ответами y
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        
        return: float
        """
        return self.oracle.func(X, y, self.w)
        
    def get_gradient(self, X, y):
        """
        Получение значения градиента функции на выборке X с ответами y
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        
        return: numpy array, размерность зависит от задачи
        """
        return self.oracle.grad(X, y, self.w)

    def get_weights(self):
        """
        Получение значения весов функционала
        """    
        return self.w
        
          
class SGDClassifier(GDClassifier):
    """
    Реализация метода стохастического градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """
    
    def __init__(self, loss_function, batch_size, step_alpha=1, step_beta=0, 
                 tolerance=1e-5, max_iter=1000, random_seed=153, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора. 
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        - 'multinomial_logistic' - многоклассовая логистическая регрессия
        
        batch_size - размер подвыборки, по которой считается градиент
        
        step_alpha - float, параметр выбора шага из текста задания
        
        step_beta- float, параметр выбора шага из текста задания
        
        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если (f(x_{k+1}) - f(x_{k})) < tolerance: то выход 
        
        
        max_iter - максимальное число итераций
        
        random_seed - в начале метода fit необходимо вызвать np.random.seed(random_seed).
        Этот параметр нужен для воспроизводимости результатов на разных машинах.
        
        **kwargs - аргументы, необходимые для инициализации
        """
        
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.kwargs = kwargs
               
    def fit(self, X, y, w_0=None, trace=False, calc_accuracy=False, log_freq=1):
        
        y = y.astype(int)
        if calc_accuracy is True:
            X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                                train_size=0.7, random_state=11)
        else:
            X_train = X
            y_train = y
        
        np.random.seed(self.random_seed)
        if trace is True:
            history = {}

        if w_0 is None:
            if self.loss_function == 'binary_logistic':
                self.w = np.random.uniform(-1, 1, X_train.shape[1])
            elif self.loss_function == 'multinomial_logistic':
                self.w = np.random.uniform(-1, 1, (np.amax(y_train) + 1, X_train.shape[1]))
        else:
            self.w = w_0

        if self.loss_function == 'binary_logistic':
            self.oracle = BinaryLogistic(**self.kwargs)
        elif self.loss_function == 'multinomial_logistic':
            self.oracle = MulticlassLogistic(**self.kwargs)
        else:
            raise TypeError("GDClassifier.fit: unknown loss_function")
        
        f = self.get_objective(X_train, y_train)
        epoch_num = 0
        if trace is True:
            history['epoch_num'] = [epoch_num]
            history['func'] = [f]
            history['time'] = [0]
            if calc_accuracy is True:
                history['accuracy'] = [np.sum(self.predict(X_test) == y_test) / len(y_test)]
            start = time.clock()
        iteration = 1    
        epoch_num_pred = epoch_num
        
        all_indexes = np.random.choice(X_train.shape[0], self.batch_size * self.max_iter)
        while iteration <= self.max_iter:
            indexes = all_indexes[(iteration-1) * self.batch_size:iteration * self.batch_size]
            self.w = (self.w - self.step_alpha / (iteration ** self.step_beta) * 
                      self.get_gradient(X_train[indexes], y_train[indexes]))
            epoch_num = self.batch_size * iteration / X_train.shape[0]
            if (epoch_num - epoch_num_pred) >= log_freq:
                f_pred = f
                epoch_num_pred = epoch_num
                f = self.get_objective(X_train, y_train)
                if trace is True:
                    history['epoch_num'].append(epoch_num)
                    history['func'].append(f)
                    history['time'].append(time.clock() - start)
                    start = time.clock()
                    if calc_accuracy is True:
                        history['accuracy'].append(np.sum(self.predict(X_test) == y_test) / 
                                                   len(y_test))
                if abs(f - f_pred) < self.tolerance:
                    break
            iteration += 1
        if trace is True:
            return history 
        else:
            return self      
