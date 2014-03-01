import logging, string, sys, glob
from gensim import corpora, models, similarities
from collections import Counter

data=open('uiuc-sports-annotations.txt')
texts=list()
for line in data:
    texts.append(line.strip().split(','))
info=open('uiuc-sports-info.txt')
label={'badminton':[0, 313], 'bocce':[314, 450], 'croquet':[451, 779], 'polo': [780, 962], 'rockclimbing': [963, 1156], 'rowing': [1157, 1412], 'sailing': [1412, 1601], 'snowboarding': [1602, 1791]}
''''
allTokens=sum(texts,[])
tokensLess=set(word for word in set(allTokens) if allTokens.count(word)<3)
texts=[[word for word in text if word not in tokensLess] for text in texts]
#build dicionary and corpus
dictionary=corpora.Dictionary(texts)
dictionary.save('dictionary.dict')
corpus=[dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('corpus.mm', corpus)
'''
#load dictionary
dictionary=corpora.Dictionary.load('dictionary.dict')
#load corpus
corpus=corpora.MmCorpus('corpus.mm')
#build lda model
lda=models.LdaModel(corpus, id2word=dictionary, num_topics=100)
#test document
#TODO: generate words here from object detection
#need generate words like following
test='stuff, parapet, tree, battledore, shuttlecock, player, sky, window'
#convert test document
bow=dictionary.doc2bow(test.lower().split(','))
vec=lda[bow]
index=similarities.MatrixSimilarity(lda[corpus])
#calculate similarities
sims=index[vec]
#sims is (document_number, document_similarity) 2-tuples
#sort based on similarities
simsLabel=list()
sims=sorted(enumerate(sims), key=lambda item: -item[1])
for s in sims:
    if s[1]<0.5:
        break
    else:
        for c in label.keys():
            begin=label[c][0]
            end=label[c][1]
            if s[0]>=begin and s[0]<=end:
                simsLabel.append(c)
print simsLabel
counts=Counter(simsLabel)
print counts.most_common()