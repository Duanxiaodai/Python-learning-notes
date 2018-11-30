
import os
import numpy as np
from utils import *
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB


# 训练
def train(normal_len, spam_len, fvs):
	train_labels = np.zeros(normal_len+spam_len)
	train_labels[normal_len:] = 1
	# SVM
	model1 = LinearSVC()
	model1.fit(fvs, train_labels)
	joblib.dump(model1, 'LinearSVC.m')
	# 贝叶斯
	model2 = MultinomialNB()
	model2.fit(fvs, train_labels)
	joblib.dump(model2, 'MultinomialNB.m')


# 测试
def test(model_path, fvs, labels):
	model = joblib.load(model_path)
	result = model.predict(fvs)
	print(confusion_matrix(labels, result))


if __name__ == '__main__':
	# Part1
	'''
	stopWords = getStopWords(txt_path='./data/stopWords.txt')
	wordsDict = wordsCount(filepath='./data/normal', stopWords=stopWords)
	wordsDict = wordsCount(filepath='./data/spam', stopWords=stopWords, wordsDict=wordsDict)
	saveDict(dict_data=wordsDict, savepath='./results.pkl')
	'''
	# Part2
	'''
	wordsDict = readDict(filepath='./results.pkl')
	wordsDict = getDictTopk(dict_data=wordsDict, topk=4000)
	saveDict(dict_data=wordsDict, savepath='./wordsDict.pkl')
	'''
	# Part3
	'''
	normal_path = './data/normal'
	spam_path = './data/spam'
	wordsDict = readDict(filepath='./wordsDict.pkl')
	normals = getFilesList(filepath=normal_path)
	spams = getFilesList(filepath=spam_path)
	fvs = []
	for normal in normals:
		fv = extractFeatures(filepath=os.path.join(normal_path, normal), wordsDict=wordsDict, fv_len=4000)
		fvs.append(fv)
	normal_len = len(fvs)
	for spam in spams:
		fv = extractFeatures(filepath=os.path.join(spam_path, spam), wordsDict=wordsDict, fv_len=4000)
		fvs.append(fv)
	spam_len = len(fvs) - normal_len
	print('[INFO]: Noraml-%d, Spam-%d' % (normal_len, spam_len))
	fvs = mergeFv(fvs)
	saveNparray(np_array=fvs, savepath='./fvs_%d_%d.npy' % (normal_len, spam_len))
	'''
	# Part4
	'''
	fvs = readNparray(filepath='fvs_7063_7775.npy')
	normal_len = 7063
	spam_len = 7775
	train(normal_len, spam_len, fvs)
	'''
	# Part5
	wordsDict = readDict(filepath='./wordsDict.pkl')
	test_normalpath = './data/test/normal'
	test_spampath = './data/test/spam'
	test_normals = getFilesList(filepath=test_normalpath)
	test_spams = getFilesList(filepath=test_spampath)
	normal_len = len(test_normals)
	spam_len = len(test_spams)
	fvs = []
	for test_normal in test_normals:
		fv = extractFeatures(filepath=os.path.join(test_normalpath, test_normal), wordsDict=wordsDict, fv_len=4000)
		fvs.append(fv)
	for test_spam in test_spams:
		fv = extractFeatures(filepath=os.path.join(test_spampath, test_spam), wordsDict=wordsDict, fv_len=4000)
		fvs.append(fv)
	fvs = mergeFv(fvs)
	labels = np.zeros(normal_len+spam_len)
	labels[normal_len:] = 1
	test(model_path='LinearSVC.m', fvs=fvs, labels=labels)
	test(model_path='MultinomialNB.m', fvs=fvs, labels=labels)