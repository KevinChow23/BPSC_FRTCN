import numpy as np
import random
import copy
'''基于Logistic回归麻雀算法(MSSA) ，write byJack旭:https://mianbaoduo.com/o/JackYM'''
'''如需其他代码请访问：链接：https://pan.baidu.com/s/1QIHWRh0bNfZRA8KCQGU8mg 提取码：1234'''
'''[1]陈刚,林东,陈飞,陈祥宇.基于Logistic回归麻雀算法的图像分割[J/OL].北京航空航天大学学报:1-14[2021-09-26].https://doi.org/10.13700/j.bh.1001-5965.2021.0268.
'''


''' 种群初始化函数 '''
def initial(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = random.random()*(ub[j] - lb[j]) + lb[j]
    
    return X,lb,ub
            
'''边界检查函数'''
def BorderCheck(X,ub,lb,pop,dim):
    for i in range(pop):
        for j in range(dim):
            if X[i,j]>ub[j]:
                X[i,j] = ub[j]
            elif X[i,j]<lb[j]:
                X[i,j] = lb[j]
    return X
    
    
'''计算适应度函数'''
import numpy as np

'''计算适应度函数'''
def CaculateFitness(X, fun):
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        a = X[i, 0]
        # 如果batch_size是NaN，使用默认值64
        b = int(X[i, 1]) if not np.isnan(X[i, 1]) else 20
        c = X[i, 2]
        # 如果kernel_size是NaN，使用默认值3
        d = int(X[i, 3]) if not np.isnan(X[i, 3]) else 3
        fitness[i] = fun(a, b, c, d)
    return fitness


'''适应度排序'''
def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness,index


'''根据适应度对位置进行排序'''
def SortPosition(X,index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i,:] = X[index[i],:]
    return Xnew

'''麻雀发现者更新'''
def PDUpdate(X, PDNumber, ST, Max_iter, dim, ub, lb, v, i, Xold):
    X_new = copy.copy(X)
    R2 = random.random()
    for j in range(PDNumber):
        if i <6 :
            if R2 < ST:
                # 改进点：逐维小孔成像反向学习
                k = 10000  # 缩放因子
                lb = np.array(lb)
                ub = np.array(ub)
                X_new[j, :] = (ub.T + lb.T) / 2 + (ub.T + lb.T) / (2 * k) - X[j, :] / k
            else:
                X_new[j, :] = X[j, :] + np.random.randn() * np.ones([1, dim])
        else:
            X1 = Xold[i - 1]
            X2 = Xold[i - 2]
            X3 = Xold[i - 3]
            X4 = Xold[i - 4]
            X5 = Xold[i - 5]
            X6 = Xold[i - 6]
            if R2 < ST:
                # 改进点：逐维小孔成像反向学习
                k = 10000  # 缩放因子
                lb = np.array(lb)
                ub = np.array(ub)
                X_new[j, :] = (ub.T + lb.T) / 2 + (ub.T + lb.T) / (2 * k) - (v*X[j, :]-0.5*v*(v-1)*X1[j, :]-v*(v-1)*(v-2)*X2[j, :]/6 + v*(v-1)*(v-2)*(v-3)*X3[j, :]/24
                              + v*(v-1)*(v-2)*(v-3)*(v-4)*X4[j, :]/120 + v*(v-1)*(v-2)*(v-3)*(v-4)*(v-5)*X5[j, :]/720 +
                              v*(v-1)*(v-2)*(v-3)*(v-4)*(v-5)*(v-6)*X6[j, :]/5040) / k
            else:
                X_new[j, :] =  (v*X[j, :]-0.5*v*(v-1)*X1[j, :]-v*(v-1)*(v-2)*X2[j, :]/6 + v*(v-1)*(v-2)*(v-3)*X3[j, :]/24
                              + v*(v-1)*(v-2)*(v-3)*(v-4)*X4[j, :]/120 + v*(v-1)*(v-2)*(v-3)*(v-4)*(v-5)*X5[j, :]/720 +
                              v*(v-1)*(v-2)*(v-3)*(v-4)*(v-5)*(v-6)*X6[j, :]/5040) + np.random.randn() * np.ones([1, dim])
    return X_new

        
'''麻雀加入者更新'''            
def JDUpdate(X,PDNumber,pop,dim):
    X_new = copy.copy(X)
    for j in range(PDNumber+1,pop):
         if j>(pop - PDNumber)/2 + PDNumber:
             X_new[j,:]= np.random.randn()*np.exp((X[-1,:] - X[j,:])/j**2)
         else:
             #产生-1，1的随机数
             A = np.ones([dim,1])
             for a in range(dim):
                 if(random.random()>0.5):
                     A[a]=-1       
         AA = np.dot(A,np.linalg.inv(np.dot(A.T,A)))
         X_new[j,:]= X[1,:] + np.abs(X[j,:] - X[1,:])*AA.T
           
    return X_new                    
            
'''危险更新'''   
def SDUpdate(X,pop,SDNumber,fitness,BestF):
    X_new = copy.copy(X)
    Temp = range(pop)
    RandIndex = random.sample(Temp, pop)
    SDchooseIndex = RandIndex[0:SDNumber]
    for j in range(SDNumber):
        if fitness[SDchooseIndex[j]]>BestF:
            X_new[SDchooseIndex[j],:] = X[0,:] + np.random.randn()*np.abs(X[SDchooseIndex[j],:] - X[1,:])
        elif fitness[SDchooseIndex[j]] == BestF:
            K = 2*random.random() - 1
            X_new[SDchooseIndex[j],:] = X[SDchooseIndex[j],:] + K*(np.abs( X[SDchooseIndex[j],:] - X[-1,:])/(fitness[SDchooseIndex[j]] - fitness[-1] + 10E-8))
    return X_new
              
    

'''改进麻雀搜索算法'''
def MSSA(pop,dim,lb,ub,Max_iter,evaluate_model):
    ST = 0.6 #预警值
    PD = 0.7 #发现者的比列，剩下的是加入者
    SD = 0.2 #意识到有危险麻雀的比重
    PDNumber = int(pop*PD) #发现者数量
    SDNumber = int(pop*SD) #意识到有危险麻雀数量
    X,lb,ub = initial(pop, dim, ub, lb) #初始化种群
    fitness = CaculateFitness(X,evaluate_model) #计算适应度值
    fitness,sortIndex = SortFitness(fitness) #对适应度值排序
    X = SortPosition(X,sortIndex) #种群排序
    GbestScore = copy.copy(fitness[0])
    GbestPositon = np.zeros([1,dim])
    GbestPositon[0,:] = copy.copy(X[0,:])
    Curve = np.zeros([Max_iter,1])
    Xold = []
    for i in range(Max_iter):
        BestF = fitness[0]

        Xold.append(X)
        dg = 0
        for j in range(pop):
            for a in range(dim):
                dg = dg + np.sqrt((GbestPositon[0, a] - X[j, a]) ** 2)
        d = np.zeros([pop])
        for ii in range(pop):
            d[ii] = 0
            for j in range(pop):
                for a in range(dim):
                    d[ii] = d[ii] + np.sqrt((X[ii, a] - X[j, a]) ** 2)
        dmin = np.min(d)
        dmax = np.max(d)
        f = (dg - dmin) / (dmax - dmin)  # 进化因子
        v = 1 / (2 * np.exp(-0.47 * f))
        # 分数阶

        # 改进点：logistic自适应因子

        ST = 0.6 * (1 / (1 + np.exp(i / 10 - 20)))
        X = PDUpdate(X, PDNumber, ST, Max_iter, dim, ub, lb, v, i, Xold)  # 发现者更新

        X = JDUpdate(X,PDNumber,pop,dim) #加入者更新
        
        X = SDUpdate(X,pop,SDNumber,fitness,BestF) #危险更新
        
        X = BorderCheck(X,ub,lb,pop,dim) #边界检测

        fitness = CaculateFitness(X,evaluate_model) #计算适应度值

        fitness,sortIndex = SortFitness(fitness) #对适应度值排序
        X = SortPosition(X,sortIndex) #种群排序
        if(fitness[0]<=GbestScore): #更新全局最优
            GbestScore = copy.copy(fitness[0])
            GbestPositon[0,:] = copy.copy(X[0,:])
        Curve[i] = GbestScore
        print(f'epoch : {i}', GbestScore, GbestPositon)
    return GbestScore,GbestPositon









