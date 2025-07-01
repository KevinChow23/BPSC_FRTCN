import numpy as np
import random
import copy
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


''' 种群初始化函数 '''
def initial(pop, dim, ub, lb,fun):
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
def CaculateFitness(X,fun):
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(X[i, 0], int(X[i, 1]), X[i, 2], int(X[i, 3]) )
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
def PDUpdate(X, PDNumber, ST, Max_iter, dim, v, i, Xold):
    X_new = copy.copy(X)
    R2 = random.random()
    for j in range(PDNumber):
        if R2 < ST:
            X_new[j, :] = X[j, :] * np.exp(-j / (np.random.random() * Max_iter))
        else:
            if i < 6:
                X_new[j, :] = X[j, :] + np.random.randn() * np.ones([1, dim])
            else:  # 引入分数阶
                X1 = Xold[i - 1]
                X2 = Xold[i - 2]
                X3 = Xold[i - 3]
                X4 = Xold[i - 4]
                X5 = Xold[i - 5]
                X6 = Xold[i - 6]
                X_new[j, :] =(v*X[j, :]-0.5*v*(v-1)*X1[j, :]-v*(v-1)*(v-2)*X2[j, :]/6 + v*(v-1)*(v-2)*(v-3)*X3[j, :]/24
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
 
'''随机游走策略 '''
def RandomWalk(Dim,max_iter,lb, ub,position,current_iter,dim):
    I=1 # 系数
    if current_iter>max_iter/10:
        I=1+100*(current_iter/max_iter)
    if current_iter>max_iter/2:
        I=1+1000*(current_iter/max_iter)
    if current_iter>max_iter*(3/4):
        I=1+10000*(current_iter/max_iter)
    if current_iter>max_iter*(0.9):
        I=1+100000*(current_iter/max_iter)
    if current_iter>max_iter*(0.95):
        I=1+1000000*(current_iter/max_iter)
        
    # 根据系数改变搜索范围，迭代次数越大，范围越小。
    lb = [lb_element / I for lb_element in lb]
    ub = [ub_element / I for ub_element in ub]

    lb = np.array(lb)
    ub = np.array(ub)

    # 进行转置
    lb = lb.T
    ub = ub.T

    # 将范围移动到目标附近
    if np.random.random()<0.5:
        lb=lb+position   
    else:
        lb=-lb+position
    
    if np.random.random()>=0.5:
        ub=ub+position
    else:
        ub=-ub+position
    
    out = np.zeros([1,dim])
    for i in range(Dim):
        A = 2*(np.random.random([max_iter,1])>0.5)-1
        X = np.cumsum(A.T)
        a = np.min(A)
        b = np.max(A)
        c = lb[i]
        d = ub[i]
        X_norm=((X-a)*(d-c))/(b-a+10E-8)+c
        out[0,i] = X_norm[current_iter]
    
    return out             

'''麻雀搜索算法'''
def RandomWalkSSA(pop,dim,lb,ub,Max_iter,evaluate_model):
    ST = 0.6 #预警值
    PD = 0.7 #发现者的比列，剩下的是加入者
    SD = 0.2 #意识到有危险麻雀的比重
    p = 0.5 #t分布变异概率
    PDNumber = int(pop*PD) #发现者数量
    SDNumber = int(pop*SD) #意识到有危险麻雀数量
    X,lb,ub = initial(pop, dim, ub, lb,evaluate_model) #初始化种群
    fitness = CaculateFitness(X,evaluate_model) #计算适应度值
    fitness,sortIndex = SortFitness(fitness) #对适应度值排序
    X = SortPosition(X,sortIndex) #种群排序
    GbestScore = copy.copy(fitness[0])
    GbestPositon = np.zeros([1,dim])
    GbestPositon[0,:] = copy.copy(X[0,:])
    Curve = np.zeros([Max_iter,1])
    Xold = []
    for i in range(Max_iter):
        factor = 1- i/Max_iter #惯性因子
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

        X = PDUpdate(X, PDNumber, ST, Max_iter, dim,
                     v, i, Xold)  # 发现者更新，引入分数阶改进

        X = JDUpdate(X,PDNumber,pop,dim) #加入者更新
        
        X = SDUpdate(X,pop,SDNumber,fitness,BestF) #危险更新
        
        X = BorderCheck(X,ub,lb,pop,dim) #边界检测     
        fitness = CaculateFitness(X,evaluate_model) #计算适应度值
        fitness,sortIndex = SortFitness(fitness) #对适应度值排序
        X = SortPosition(X,sortIndex) #种群排序
        #随机游走策略对最有位置进行扰动
        Temp = RandomWalk(dim,Max_iter,lb, ub,X[0,:],i,dim)
        Temp = BorderCheck(Temp,ub,lb,1,dim)
        fitTemp = evaluate_model(Temp[0,0], Temp[0,1], Temp[0,2],Temp[0,3] )
        if fitTemp < fitness[0]:
            fitness[0] = copy.copy(fitTemp)
            X[0] = copy.copy(Temp[0])  
        if(fitness[0]<=GbestScore): #更新全局最优
            GbestScore = copy.copy(fitness[0])
            GbestPositon[0,:] = copy.copy(X[0,:])
        Curve[i] = GbestScore
        print(f'epoch : {i}', GbestScore, GbestPositon)
    return GbestScore,GbestPositon









