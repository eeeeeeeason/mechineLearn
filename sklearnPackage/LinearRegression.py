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
        
    def predict(self, X_predict):
        """给定待预测向量，得出结果"""
        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        return X_b.dot(self._theta)

    def score(self, X_test, y_test):
        """计算该模型的准确度也称其为r2score"""
        y_predict = self.predict(X_test)
        return 1 - mean_squared_error(y_test, y_predict) / np.var(y_test)
