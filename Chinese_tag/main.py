# -*- coding: utf-8 -*-
import random
from trainTag import trainTag
from testTag import testTag

trainFile = 'train_utf16.tag'
testFile = 'test_utf16.tag'

if __name__ == '__main__':
    trainSents = []
    testSents = []
    f = open(trainFile, 'r', encoding='utf_16_le')
    for line in f:
        line = line.strip()
        if len(line) == 0:
            continue
        r = random.random()
        if r < 0.7:
            trainSents.append(line)
        else:
            testSents.append(line)
    f.close()

    tr = trainTag(trainSents)
    tr.train()
    ts = testTag(testSents, tr.wordPosFreq, tr.posFreq, tr.posTransPro, tr.wordPosHeadPro, 'test_output.tag')
    ts.test()