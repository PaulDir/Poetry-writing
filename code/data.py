import tensorflow as tf
import numpy as np
import collections

from config import *

class Poem(object):
	def __init__(self):
		self._batch_size = batch_size
		self._batchX = []
		self._batchY = []
		self._offset = 0

		poems = []
		file = open(filename, "r",encoding='utf-8')

		for line in file:
			title, author, poem = line.strip().split("::") 
			poem = poem.replace(" ","")
			if '_' in poem or '《' in poem or '[' in poem or '(' in poem or '（' in poem:
				continue
			poem = '[' + poem + ']'
			poems.append(poem)

		wordFreq = collections.Counter()
		for poem in poems:
			wordFreq.update(poem)

		wordFreq[" "] = -1
		wordPairs = sorted(wordFreq.items(), key = lambda x: -x[1])
		words,freq = zip(*wordPairs)
		self.words = words
		self.wordToid = dict(zip(words,range(len(words))))
		self.idToword = dict(zip(self.wordToid.values(), self.wordToid.keys()))
		poemsVector = [[self.wordToid[word] for word in poem]for poem in poems]
		n = int(len(poemsVector) *ratio)
		self.trainVector = poemsVector[:n]
		self.testVector = poemsVector[n:]
		self.tranVector_size = n
		self.testVector_size = len(poemsVector) - n
		self.tranSet_batch_num = (n-1) // batch_size

		self.generateBatch()

	def data_set(self):
		dictionary = self.wordToid
		reversed_dictionary = self.idToword
		return dictionary, reversed_dictionary

	def generateBatch(self, isTrain=True):
		#padding length to batchMaxLength
		if isTrain:
			poemsVector = self.trainVector
		else:
			poemsVector = self.testVector

		random.shuffle(poemsVector)
		batchNum = (len(poemsVector) - 1) // batch_size
		x_data = []
		y_data = []
		#create batch
		for i in range(batchNum):
			batch = poemsVector[i * batch_size: (i + 1) * batch_size]
			maxLength = max([len(vector) for vector in batch])
			temp = np.full((batch_size, maxLength), self.wordToid[" "], np.int32) # padding space
			for j in range(batch_size):
				temp[j, :len(batch[j])] = batch[j]
			x_data.append(temp)
			temp2 = np.copy(temp) 
			temp2[:, :-1] = temp[:, 1:]
			y_data.append(temp2)
		self._batchX = x_data
		self._batchY = y_data

	def next(self):
		x_data = self._batchX[self._offset]
		y_data = self._batchY[self._offset]
		self._offset = (self._offset + 1) % self.tranSet_batch_num
		return x_data, y_data