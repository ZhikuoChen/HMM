#-*-coding:utf-8
import sys
#import math
#将一个字作为隐性状态，将由此字构成的词作为显性状态
A_dic = {}   #表示状态转移矩阵
B_dic = {}   #发射概率矩阵
Count_dic = {}     #用于计数的字典
Pi_dic = {}  #初始状态矩阵
state_list = ['B','M','E','S']   #B（开头）,M（中间),E(结尾),S(Separate 独立成词）
def init():
    #将A,B,Pi字典都初始化为0
    for state in state_list:
        A_dic[state] = {}
        for state1 in state_list:
            #将状态转移矩阵的初值设为0
            A_dic[state][state1] = 0.0    #A_dic为二维字典
    for state in state_list:
        #设初始状态的各类初值都为0，Pi_dic为一维字典
        Pi_dic[state] = 0.0
        B_dic[state] = {}   #B_dic为二维字典
        Count_dic[state] = 0

def getList(input_str):
    output_str = []
    #如果一个词的长度为1即由一个字构成，那么此词为单独成词
    if len(input_str) == 1:
       output_str.append('S')
    #如果一个词的长度为2即由两个字构成，那么此词为B和E构成。
    elif len(input_str) == 2:
         output_str = ['B','E']
    else:
        #如果一个词的长度大于2即由多个字构成，那么此词为B,M,...M,E构成。
        M_num = len(input_str) -2
        M_list = ['M'] * M_num  #表示中间词(M)的数目
        output_str.append('B')
        output_str.extend(M_list)
        output_str.append('S')
    return output_str

def CalProb(train_set):
    File = open(train_set,'r', encoding='utf-8')
    init()
    line_num = -1
    char_set = set()
    for line in File:
        line_num += 1
        #.strip()用于移除字符串头尾指定的字符（默认为空格）
        line = line.strip()
        if not line:
           continue
        char_list = []
        for i in range(len(line)):
            if line[i] == " ":
               continue
            char_list.append(line[i])   
        #char_set存储的是字
        char_set = char_set | set(char_list)   #求char_set集合和char_list的合集
        #.split()将一个字符串分裂成多个字符串组成的列表。
        word_list = line.split(" ")   #word_list存储的是词
        char_state = []
        for item in word_list:
            #表示将每个构成词的每个状态变成['B','M','E','S']之一，添加到line_state中
            char_state.extend(getList(item))
#        print(line)    #打印当前句
#        print(char_list)  #打印当前句的每个字
#        print(char_state)  #打印当前句的每个字的状态
        #此时字的状态数应和字的总数相同，否则报错
        if len(char_list) != len(char_state):
            #line.endoce("utf-8",'ignore')表示把其他编码转成utf8的格式，并忽略非法字符。
            print("[line_num = %d][line = %s]" % (line_num, line.endoce("utf-8",'ignore')),
                  file=sys.stderr)
        else:
            for i in range(len(char_state)):
                #初始化状态 ,并将此种状态的计数值加1
                if i == 0:
                    Pi_dic[char_state[0]] += 1
                else:
                    #从前一状态M转移到后一状态N的次数加1
                    A_dic[char_state[i-1]][char_state[i]] += 1
                    
                    #将一个字的状态['B'/'M'/'E'/'S']作为隐性状态，属于某种状态下，该字出现的次数作为
                    #隐性状态。
                    if not char_list[i] in B_dic[char_state[i]]:
                        B_dic[char_state[i]][char_list[i]] = 0.0
                    else:
                        B_dic[char_state[i]][char_list[i]] += 1
                #将此字属于某种状态的计数值加1
                Count_dic[char_state[i]] += 1
    #输出字的总数
    print('训练集共有%d个字' %len(char_set))
    for key in Pi_dic:
        '''
        if Pi_dic[key] != 0:
            Pi_dic[key] = -1*math.log(Pi_dic[key] * 1.0 / line_num)
        else:
            Pi_dic[key] = 0
        '''
        #求初始时，各以'B','M','E','S'四种词开头的句子的概率
        Pi_dic[key] = Pi_dic[key] * 1.0 / line_num

    for key1 in A_dic:
        for key2 in A_dic[key1]:
            '''
            if A_dic[key1][key2] != 0:
                A_dic[key1][key2] = -1*math.log(A_dic[key1][key2] / Count_dic[key1])
            else:
                A_dic[key1][key2] = 0
            '''
            #从key1状态转移到key2状态的概率为从key1状态转移到key2状态的字出现的次数除以key1状态
            #出现的总次数
            A_dic[key1][key2] = A_dic[key1][key2] / Count_dic[key1]
    for key in B_dic:
        for word in B_dic[key]:
            '''
            if B_dic[key][word] != 0:
                B_dic[key][word] = -1*math.log(B_dic[key][word] / Count_dic[key])
            else:
                B_dic[key][word] = 0
            '''
            #某一字作为'B','M','E','S'四种状态的概率等于该字作为此状态出现的次数除以该种状态出现的
            #总次数
            B_dic[key][word] = B_dic[key][word] / Count_dic[key]
    #返回数据库的初始状态概率，状态转移概率，发射概率
    return(Pi_dic,A_dic,B_dic)
