'''
用HMM实现汉语分词
隐状态为:['B','M','E','S']
状态转移概率为由一种状态转移到另一种状态的概率，如由'B'转移到'E'的概率
发射概率为此种状态对应的每个汉字的概率:
'''
import CalProbability as cp
import time
def Viterbi(observe, states, start_p, trans_p, emit_p):
    V = [{}]      #路径概率表V[位置][隐状态] = 概率
    path = {}     #为一字典
    for y in states: #初始化
        #在位置0，以y状态为末尾的状态序列的最大概率
        #.get表示获得给定键值对应的值，若不存在，则返回0
        V[0][y] = start_p[y] * emit_p[y].get(observe[0],0)
        path[y] = [y]
    for t in range(1,len(observe)):
        V.append({})
        newpath = {}
        for y in states:
              #从y0 -> y状态的递归
            (prob,state) = max([(V[t-1][y0] * trans_p[y0].get(y,0) * emit_p[y].get(observe[t],0) ,y0) for y0 in states if V[t-1][y0]>0])
            V[t][y] =prob
            newpath[y] = path[state]+[y]  
         #记录状态序列
        path = newpath
    print(path)
    #在最后一个位置，以y状态为末尾的状态序列的最大概率
    (prob, state) = max([(V[len(observe) - 1][y], y) for y in states])
    return (prob, path[state])        #返回概率和状态序列
def printResult(sen,state):
    wordList=[]
    string=''  #定义一个空字符串
    for i in range(len(state)):
        if state[i]=='B' or state[i]=='M':
             string=string+sen[i] 
        else:
            string=string+sen[i]                
            wordList.append(string)
            string=''
    return wordList
        
if __name__ == "__main__":
    start=time.clock()#程序开始时间
    TrainData= "RenMinData.txt_utf8"
    TestSentence = u"长春市长春节讲话。"
    #    TestSentence = u"他说的确实在理."
    #    TestSentence = u"无鸡鸭亦可，无鱼肉也可，白菜豆腐不能少。"
    state_list = ['B','M','E','S']   #B（开头）,M（中间),E(结尾),S(Separate 独立成词）
    print('程序正在运行！')
    prob_start, prob_trans, prob_emit=cp.CalProb(TrainData)
    print('状态转移概率为：')
    print(prob_trans)
    print("发射概率:")
    print(prob_emit)
    prob,testCharState_list=Viterbi(TestSentence,state_list, prob_start, prob_trans, prob_emit)
    
    print('测试句子为：',TestSentence)
    word_list=printResult(TestSentence,testCharState_list)
    print('分词结果为：',word_list)
    print('测试句子中每个字的状态为：',testCharState_list)
    
    end =time.clock() #程序结束时间
    print('程序运行了%d秒' %(end-start))