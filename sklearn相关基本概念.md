### 训练数据集与测试数据集
- 通常测试模型与算法是否准确，我们会将实际数据拆分为两部分，一部分拿来生成训练模型，另一部分则是检测该模型的准确度叫测试数据集
- knn中的模型即是实际数据，以鸢尾花为例，我们可以拿到乱序后的鸢尾花数据80%做为模型，剩余20%进行knn算法分类，检测这个算法是否合理

```
from sklearn.model_selection import train_test_split  // 切割数据，分测试数据集、训练数据集
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,0.2,666)
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train, y_train)
y_predict = knn_clf.predict(X_test)
from sklearn.metrics import accuracy_score  // 计算该模型的准确性
accuracy_score(y_test, y_predict)
knn_clf.score(X_test, y_test)
```

- 超参数，需要在算法运行前获取，跟不同环境类型有关，gridsearch可获取最佳超参数[超参数的网格交叉校验计算优化及使用案例](https://github.com/eeeeeeeason/mechineLearn/blob/master/%E7%BD%91%E6%A0%BC%E6%90%9C%E7%B4%A2-%E8%B6%85%E5%8F%82%E6%95%B0%E4%BC%98%E5%8C%96gridsearch)
