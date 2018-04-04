import numpy as np


class MulticlassStrategy:   
    def __init__(self, classifier, mode, **kwargs):
        """
        Инициализация мультиклассового классификатора
        
        classifier - базовый бинарный классификатор
        
        mode - способ решения многоклассовой задачи,
        либо 'one_vs_all', либо 'all_vs_all'
        
        **kwargs - параметры классификатор
        """
        self.cl = classifier
        self.args = kwargs
        self.mode = mode        
        
    def fit(self, X, y):
        """
        Обучение классификатора
        """
        self.cls = []
        if self.mode == 'one_vs_all':
            for i in range(np.amax(y) + 1):
                self.cls.append(self.cl(**self.args))
                y_i = np.where(y == i, 1, -1)
                self.cls[i].fit(X, y_i)
            return self
        elif self.mode == 'all_vs_all':
            for j in range(np.amax(y) + 1):
                for s in range(j):
                    cl = self.cl(**self.args)
                    y_sj = np.any([y == s, y == j], axis=0)
                    cl.fit(X[y_sj], np.where(y[y_sj] == s, 1, -1))
                    self.cls.append(cl)
        else:
            raise TypeError("MulticlassStrategy.fit: unknown mode")
        
    def predict(self, X):
        """
        Выдача предсказаний классификатором
        """
        if self.mode == 'one_vs_all':
            results = np.array([cl.predict_proba(X)[:, 1] for cl in self.cls])
            return np.argmax(results, axis=0)
        elif self.mode == 'all_vs_all':
            results = np.array([np.where(self.cls[s + j - 1].predict(X) == 1, s, j) 
                                for j in range(np.amax(y + 1)) for s in range(j)])
            return np.array([np.bincount(preds).argmax() for preds in results.T])
