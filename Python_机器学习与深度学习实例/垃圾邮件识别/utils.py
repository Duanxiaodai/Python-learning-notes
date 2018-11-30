
import re
import os
import jieba
import pickle
import numpy as np


# 获取停用词列表
def getStopWords(txt_path='./data/stopWords.txt'):
	stopWords = []
	with open(txt_path, 'r') as f:
		for line in f.readlines():
			stopWords.append(line[:-1])
	return stopWords


# 把某list统计进dict
def list2Dict(wordsList, wordsDict):
	for word in wordsList:
		if word in wordsDict.keys():
			wordsDict[word] += 1
		else:
			wordsDict[word] = 1
	return wordsDict


# 获取文件夹下所有文件名
def getFilesList(filepath):
	return os.listdir(filepath)


# 统计某文件夹下所有邮件的词频
def wordsCount(filepath, stopWords, wordsDict=None):
	if wordsDict is None:
		wordsDict = {}
	wordsList = []
	filenames = getFilesList(filepath)
	for filename in filenames:
		with open(os.path.join(filepath, filename), 'r') as f:
			for line in f.readlines():
				# 过滤非中文字符
				pattern = re.compile('[^\u4e00-\u9fa5]')
				line = pattern.sub("", line)
				words_jieba = list(jieba.cut(line))
				for word in words_jieba:
					if word not in stopWords and word.strip != '' and word != None:
						wordsList.append(word)
		wordsDict = list2Dict(wordsList, wordsDict)
	return wordsDict


# 保存字典类型数据
def saveDict(dict_data, savepath='./results.pkl'):
	with open(savepath, 'wb') as f:
		pickle.dump(dict_data, f)


# 读取字典类型数据
def readDict(filepath):
	with open(filepath, 'rb') as f:
		dict_data = pickle.load(f)
	return dict_data


# 对输入的字典按键值排序(降序)后返回前topk组数据
def getDictTopk(dict_data, topk=4000):
	data_list = sorted(dict_data.items(), key=lambda dict_data: -dict_data[1])
	data_list = data_list[:topk]
	return dict(data_list)


# 提取文本特征向量
def extractFeatures(filepath, wordsDict, fv_len=4000):
	fv = np.zeros((1, fv_len))
	words = []
	with open(filepath) as f:
		for line in f.readlines():
			pattern = re.compile('[^\u4e00-\u9fa5]')
			line = pattern.sub("", line)
			words_jieba = list(jieba.cut(line))
			words += words_jieba
		for word in set(words):
			for i, d in enumerate(wordsDict):
				if d[0] == word:
					fv[0, i] = words.count(word)
	return fv


# 合并特征向量
def mergeFv(fvs):
	return np.concatenate(tuple(fvs), axis=0)


# 保存np.array()数据
def saveNparray(np_array, savepath):
	np.save(savepath, np_array)


# 读取np.array()数据
def readNparray(filepath):
	return np.load(filepath)