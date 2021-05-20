from scipy import stats
import numpy as np
from random import shuffle, uniform;



xk = np.arange(7)  # 所有可能的取值
print(xk)  # [0 1 2 3 4 5 6]
xk = [0,1,2,3,4,5,6]
pk = (0.1, 0.2, 0.3, 0.1, 0.1, 0.0, 0.2)  # 各个取值的概率
custm = stats.rv_discrete(name='custm', values=(xk, pk))

X = custm.rvs(size=20)
print(X)