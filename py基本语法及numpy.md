### 模块导入，使用场景juptyer
  - 根据文件名.函数可以直接调用？？？？太简单了吧。。
  ```
    import ml.test
    %run test
    import numpy as np
  ```

### numpy生成矩阵
  - np.full(fill_value=666, shape=(3,5)) // 生成3*5全部为666（默认浮点类型矩阵），如果要规定数据类型新增dtype=int balbalbal
  
### arange步长，排序，乱序
  - x = np.arange(0, 20, 2) // array([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18])
  - np.random.shuffle(x) // 乱序
  - np.sort(x)
  
### linspace等差数列
  - np.linspace(0,20,10) //0开始，20结束，10为一共个数  array([ 0.        ,  2.22222222,  4.44444444,  6.66666667,  8.88888889,
       11.11111111, 13.33333333, 15.55555556, 17.77777778, 20.        ])

### random 获取随机数，随机列表
  - np.random.randint(0,10) // 生成0-10的整数d
  - np.random.randint(4,8, size=(3,5)) //4-8的随机数生成3*5矩阵
  - np.random.normal(10, 100, (3,5)) // 生成均值为10方差为100的3*5矩阵 

## 数组方法
  ### 基本方法
  - X = np.arange(10).reshape(2,5) // 重新生成矩阵
  - X.ndim   2 // 获取维度
  - X.shape  (3,5)
  - X.size   10
  - X[0,0] = X[0][0]
  - X[:5] 前5各元素
  - X[:2, :3] 前两行的前3列
  - X[:2, ::2] 前两行步长2
  - X[:,0]
  - subx = X[:2,:2].copy() // 数组的裁剪是引用类型，修改时会导致原数组变化
  ### 合并及拆分 x = np.array([1,2,3,4,5]) y = np.array([3,2,1]) A [[3,2,1],[1,2,3]] B [6,6,6] C [[6],[6]]
  - np.concatenate([x,y])   array([1,2,3,3,2,1])
  - np.vstack(A, B) // 垂直方向智能合并等价np.concatenate([A,B.reshape(1,-1)])  [[3,2,1],[1,2,3],[6,6,6]]
  - np.hstack([A,B])  [[3,2,1,6],[1,2,3,6]]   // 水平方向合并
  - np.split(x,[1,3]) [1,3]代表切割次数可根据索引分为多个新数组
  - np.vsplit(x,[2]) 水平切割第二个之后割掉
  - np.hsplit(x,[2]) 垂直
  - np.hstack([A,B])  [[3,2,1,6],[1,2,3,6]]   // 水平方向合并内
  - np.hstack([A,B])  [[3,2,1,6],[1,2,3,6]]   // 水平方向合并
  ### 矩阵的逆， 与矩阵左乘，右乘都能得到对角线为单位1的结果，只有方阵才有逆，否则都是伪逆矩阵
  - np.linalg.inv(A) // A的逆
  - pinvx = np.linalg.pinv(X) // A的伪逆
  ### 获取方差，标准差，最大值，最大值索引，最小值，最小值索引
  - np.random.normal(10, 100,size=100000) // 生成均值为10方差为100,100000个数合并的数组
  - np.argmin(x)， np.argmax // 最小/大值索引
  - np.max(x) min // 最大值
  - np.mean(x) // 平均
  - np.std(x) // 方差
  
  
