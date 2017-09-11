import numpy as np
class MyHmm(): # base class for different HMM models
    def __init__(self, State,A,B,Pi,Obs):
        self.A = A
        self.states = State 
        self.B = B
        self.Pi = Pi
        self.Obs=Obs
#此算法用于求解HMM的估计问题：给定一个观察序列和模型MU=(A,B,PI)，如何快速地计算出给定模型mu情况下，
#观察序列O的概率，即P(Obs|MU)      
    #前向算法：1.初始化，2：归纳计算，3：求和终结
    def Forward(self):
        alpha = [{}]     
        # Initialize base cases (t == 0)
        for y in self.states:
            alpha[0][y] = self.Pi[y] * self.B[y][self.Obs[0]]
        # Run Forward algorithm for t > 0
        for t in range(1, len(self.Obs)):
            alpha.append({})     
            for y in self.states:
                alpha[t][y] = sum((alpha[t-1][y0] * self.A[y0][y] *
                                        self.B[y][self.Obs[t]]) for y0 in self.states)
        #求和终结
        prob_sum=0
        for s in self.states:
            prob_sum+=alpha[len(self.Obs) - 1][s]
        return prob_sum,alpha
 
    def Backward(self):
        beta = [{} for t in range(len(self.Obs))]
        T = len(self.Obs)
        # Initialize base cases (t == T)
        for y in self.states:
            beta[T-1][y] = 1 #将最后时刻的beta值初始化为1
 #要习惯这种编程方式，共有三层循环。外层为t的循环，中间层为y的循环，内层为y1的循环
        for t in reversed(range(T-1)):
            for y in self.states:
                beta[t][y] = sum((beta[t+1][y1] * self.A[y][y1] * 
                                      self.B[y1][self.Obs[t+1]]) for y1 in self.states)
        #求和终结
        prob=0
        for state in self.states:
            prob+=self.Pi[state]* self.B[state][self.Obs[0]] * beta[0][state]                                                              
#        prob = sum((self.Pi[y]* self.B[y][self.Obs[0]] * beta[0][y]) for y in self.states)
        return prob,beta
#此算法用于求解HMM的参数估计问题：给定一个观察序列O=O1O2...Ot和模型MU=(A,B,PI)，如何根据最大似然估计
#来求模型的参数值？即如何调节模型mu=(A,B,Pi)的参数，使得P(O|mu)最大。  
    def ForwardBackward(self): # returns model given the initial model and observations        
        gamma = [{} for t in range(len(self.Obs))] # 需要跟踪在一个时间t找到所有i和全部t的状态i
        zeta = [{} for t in range(len(self.Obs) - 1)]  # this is needed to keep track of finding a state i at a time t and j at a time (t+1) for all i and all j and all t
        # 得到alpha和beta表的计算值
        obs_prob,alpha=self.Forward()
        p,beta=self.Backward()
        #如果观察值概率小于0，则报错
        if obs_prob <= 0:
            raise ValueError("P(O | lambda) = 0. Cannot optimize!")
        #EM算法：由 E-步骤 和 M-步骤 组成
        #E-步骤：计算期望值zeta和gamma
        for t in range(len(self.Obs)):
            for y in self.states:
                gamma[t][y] = (alpha[t][y] * beta[t][y]) / obs_prob
                if t == 0:
                   self.Pi[y] = gamma[t][y]
                #compute zi values up to T - 1
                if t == len(self.Obs) - 1:
                   continue
                zeta[t][y] = {}
                for y1 in self.states:
                    zeta[t][y][y1] = alpha[t][y] * self.A[y][y1] * self.B[y1][self.Obs[t + 1]] * beta[t + 1][y1] / obs_prob
        #M-步骤：重新估计参数Pi,A,B
        for y in self.states:
            for y1 in self.states:
                # 计算新的Aij
                numA = sum([zeta[t][y][y1] for t in range(len(self.Obs) - 1)]) #
                denA= sum([gamma[t][y] for t in range(len(self.Obs) - 1)])
                self.A[y][y1] = numA/denA
        for y in self.states:
            for k in self.B[self.states[0]].keys():   
                numB = 0.0
                for t in range(len(self.Obs)):
                    if self.Obs[t] == k :
                        numB += gamma[t][y]                 
                denB= sum([gamma[t][y] for t in range(len(self.Obs))])
                self.B[y][k] = numB/denB
        #返回A,B,Pi
        return self.A,self.B,self.Pi
    # 打印路径概率表
    def print_path(self,V):
        print('从第一天到第五天，每天是Rainy和Sunny的概率：')
        for y in V[0].keys():
            print(y)
            for t in range(len(V)):
                print("%0.5f" % V[t][y])
        print()
#此算法用于求解HMM的序列问题：给定一个观察序列O=O1O2...Ot和模型MU=(A,B,PI)，如何快速有效地选择在
#一定意义下“最优的”状态序列Q=q1q2...qt,使得该状态序列最好地解释观察序列。
#观察序列O的概率，即P(Obs|MU)     
    #维特比算法：1.初始化，2.归纳计算，3.终结，4.路径回溯        
    def Viterbi(self):
        """
        :param Obs:观测序列
        :param states:隐状态
        :param Pi:初始概率（隐状态）
        :param A:转移概率（隐状态）
        :param B: 发射概率（隐状态表现为显状态的概率）
        :return:
        """
        # 路径概率表delta[时间][隐状态] = 概率
        delta = [{}]
        path = {}     
        # 初始化初始状态 (t = 0)
        for y in self.states:
            delta[0][y] = self.Pi[y] * self.B[y][self.Obs[0]]
            path[y] = [y]
     
        # Run Viterbi for t > 0
        for t in range(1, len(self.Obs)):
            delta.append({})
            newpath = {}     
            for y in self.states:
                #每次循环时，y0与计算的概率值构成一个列表，然后求整个循环概率值的最大值，并求出对应的
                #状态，作为路径
                (prob, state) = max((delta[t-1][y0] * self.A[y0][y] * self.B[y][self.Obs[t]], y0) for y0 in self.states)
                delta[t][y] = prob
                newpath[y] = path[state] + [y]     
            # Don't need to remember the old paths
            path = newpath
        self.print_path(delta)
        n = 0       #如果仅观察到一个元素，则在初始化值中寻求最大值
        if len(self.Obs)!=1:
            n = t
        (prob, state) = max((delta[n][y], y) for y in self.states)
        return (prob, path[state])
        #另一种方式
#        save=[]
#        for state in self.states:
#            save.append(delta[len(self.Obs) - 1][state])
#        prob = max(save)
#        label=np.argmax(save)
#        return (prob, path[self.states[label]])

if __name__ == "__main__":
   #State为状态的集合
   State=('Sunny','Rainy')
   #Pi为初始状态的概率分布
   Pi={'Rainy': 0.6, 'Sunny': 0.4}
   #A为状态转移概率
   A={'Rainy' : {'Rainy': 0.7, 'Sunny': 0.3},'Sunny' : {'Rainy': 0.4, 'Sunny': 0.6}}
   #B为符号发射概率
   B={'Rainy' : {'walk': 0.1,'read':0.2,'shop': 0.2, 'clean': 0.5},
       'Sunny' : {'walk': 0.4, 'shop': 0.3,'read':0.1, 'clean': 0.2}}
   #O为观察序列
   O=['clean', 'shop','read', 'walk','shop']
   #    O=['clean','shop']
   hmm=MyHmm(State,A,B,Pi,O)
   pro,TransPath=hmm.Viterbi()
   #调用前向算法计算观测值的发生概率
   ObsProb,y=hmm.Forward()
   print('在已知模型A,B,Pi的条件下，观察到序列%s->%s->%s->%s->%s的概率为' 
         %(O[0],O[1],O[2],O[3],O[4]),round(ObsProb,7))
   #输出隐状态转移的最可能过程
   print('在观察到序列%s->%s->%s->%s->%s的情况下，天气状态的转移过程最可能为%s->%s->%s->%s->%s'
          %(O[0],O[1],O[2],O[3],O[4],TransPath[0],TransPath[1],TransPath[2],TransPath[3],
            TransPath[4]))
   randA={'Rainy' : {'Rainy': 0.23, 'Sunny': 77},'Sunny' : {'Rainy': 0.64, 'Sunny': 0.36}}
   randB={'Rainy' : {'walk': 0.4,'read':0.2,'shop': 0.3, 'clean': 0.1},
       'Sunny' : {'walk': 0.2, 'shop': 0.3,'read':0.1, 'clean': 0.4}}
   randPi={'Rainy': 0.5, 'Sunny': 0.5}
   hmm1=MyHmm(State,randA,randB,randPi,O)
   EstimateA,EstimateB,EstimatePi=hmm1.ForwardBackward()
   lenPi=len(EstimatePi)
   for key1 in sorted(EstimateA.keys()):
       for key2 in EstimateA[key1].keys():
           EstimateA[key1][key2]=round(EstimateA[key1][key2],2)
   for key1 in sorted(EstimateB.keys()):
       for key2 in EstimateB[key1].keys():
           EstimateB[key1][key2]=round(EstimateB[key1][key2],2)
   for key in EstimatePi.keys():
       EstimatePi[key]=round(EstimatePi[key],2)
   print(EstimateA)
   print(EstimateB)
   print(EstimatePi)   
