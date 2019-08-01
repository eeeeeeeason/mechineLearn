### 模块导入，使用场景juptyer
  - 根据文件名.函数可以直接调用？？？？太简单了吧。。
  ```
    import ml.test
    %run test
    import numpy as np
  ```

### numpy生成矩阵
  - np.full(fill_value=666, shape=(3,5)) // 生成3*5全部为666（默认浮点类型矩阵），如果要规定数据类型新增dtype=int balbalbal
  
### arange步长
  - np.arange(0, 20, 2) // array([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18])

### linspace等差数列
  - np.linspace(0,20,10) //0开始，20结束，10为一共个数  array([ 0.        ,  2.22222222,  4.44444444,  6.66666667,  8.88888889,
       11.11111111, 13.33333333, 15.55555556, 17.77777778, 20.        ])

### random 获取随机数，随机列表
  - np.random.randint(0,10) // 生成0-10的整数d
  - np.random.randint(4,8, size=(3,5)) //4-8的随机数生成3*5矩阵
  - np.random.normal(10, 100, (3,5)) // 生成均值为10方差为100的3*5矩阵
