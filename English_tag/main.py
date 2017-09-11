# -*- coding: utf-8 -*-
import codecs
import random
from trainTag import trainTag
from testTag import testTag

trainFile = 'conll2000train.txt'
testFile = 'conll2000test.txt'

if __name__ == '__main__':
    trainSents = []
    testSents = []
    f = codecs.open(trainFile, 'r', encoding='utf_8')
    # length = 0
    pairList = []
    for line in f:
        # length += 1
        line = line.strip()
        if len(line) != 0:
            l = line.split(' ')
            pairList.append( [l[0], l[1]] )
        else:
            r = random.random()
            trainSents.append(pairList)
            pairList = []
    f.close()

    f = codecs.open(testFile, 'r', encoding='utf_8')
    pairList = []
    for line in f:
        line = line.strip()
        if len(line) != 0:
            l = line.split(' ')
            pairList.append( [l[0], l[1]] )
        else:
            testSents.append(pairList)
            pairList = []
    f.close()

    tr = trainTag(trainSents)
    tr.train()
    ts = testTag(testSents, tr.wordPosFreq, tr.posFreq, tr.posTransPro, 
                                    tr.wordPosHeadPro, 'test_result.txt')
    ts.test()