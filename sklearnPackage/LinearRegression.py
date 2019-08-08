from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
class LinearRegression:
    def __init__(self):
        """coef 对应一维的a, interception对应b , 多维时 coef代表向量[theta1,....thetan]除了theta0, interception代表theta0"""
        self.coef_ = None
        self.interception_ = None
        self._theta = None
    
    def fit_normal(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        """生成一列为1的向量，长度与训练数据相同进行合并，目的是为了凑theta0的X，使每一项都存在X"""
        X_b = np.hstack([np.ones((len(X_train),1)), X_train]) 
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]
        
        return self
    
    def fit_gd(self, X_train, y_train, eta=0.01, n_iters=1e4):
        """根据训练数据集，使用梯度下降法训练"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"
        
        def J(theta, X_b, y):
            try:
                return np.sum((y - X_b.dot(theta)) ** 2) / len(X_b)
            except:
                return float('inf')

        def dJ(theta, X_b, y):
            #res = np.empty(len(theta))
            #res[0] = np.sum(X_b.dot(theta) - y)
            #for i in range(1,len(theta)):
            #    res[i] = (X_b.dot(theta) - y).dot(X_b[:,i])
            #return res * 2 / len(X_b)
            return X_b.T.dot(X_b.dot(theta) - y) * 2. / len(X_b)
            
        def gradient_descent(X_b, y,initial_theta, eta, n_iters=1e4, epsilon=1e-8):
            theta = initial_theta
            i_iter = 0
            """限制循环最多执行次数"""
            while i_iter < n_iters:
                gradient = dJ(theta, X_b, y)
                last_theta = theta
                theta = theta - eta * gradient
                if (abs(J(theta, X_b, y) - J(last_theta, X_b, y)) < epsilon):
                    break
                i_iter += 1
            return theta
        
        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        initial_theta = np.zeros(X_b.shape[1])
        self._theta = gradient_descent(X_b, y,initial_theta, eta, n_iters)
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        
    def predict(self, X_predict):
        """给定待预测向量，得出结果"""
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        """计算该模型的准确度也称其为r2score"""
        y_predict = self.predict(X_test)
        return 1 - mean_squared_error(y_test, y_predict) / np.var(y_test)
