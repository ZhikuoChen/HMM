# -*- coding: utf-8 -*-
import random
from trainNer import trainNer
from testNer import testNer

trainFile = 'train_utf16.ner'
testFile = 'test_utf16.ner'  #此词汇未被标注

if __name__ == '__main__':
    trainSents = []
    testSents = []
    f = open(trainFile, 'r', encoding='utf_16_le')
    pairList = []
    for line in f:
        line = line.strip()
        if len(line) != 0:
            l = line.split(' ')
            pairList.append([l[0], l[1]])
        else:
            r = random.random()
            #将70%数据用于训练，30%数据用于测试
            if r < 0.7:
                trainSents.append(pairList)
            else:
                testSents.append(pairList)
            pairList = []
    f.close()

    tr = trainNer(trainSents)
    tr.train()
    ts = testNer(testSents, tr.wordPosFreq, tr.posFreq, tr.posTransPro, tr.wordPosHeadPro, 'output.ner')
    ts.test()
    
    