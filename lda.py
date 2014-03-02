import logging, string, sys, glob
from gensim import corpora, models, similarities
from collections import Counter
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

data=open('train.txt')
texts=list()
for line in data:
    texts.append(line.strip().split(','))
info=open('label.txt')
label=list()
for line in info:
    label.append(line.strip())

allTokens=sum(texts,[])
tokensLess=set(word for word in set(allTokens) if allTokens.count(word)<3)
texts=[[word for word in text if word not in tokensLess] for text in texts]
#build dicionary and corpus
dictionary=corpora.Dictionary(texts)
dictionary.save('dictionary.dict')
corpus=[dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('corpus.mm', corpus)

#load dictionary
#dictionary=corpora.Dictionary.load('dictionary.dict')
#load corpus
#corpus=corpora.MmCorpus('corpus.mm')
#build lda model
lda=models.LdaModel(corpus, id2word=dictionary, num_topics=100)
#test document
''''
test='34,248,248,248,179,17,17,34,248,138,138,248,122,122,256,256,256,34,64,223,121,121,122,122,179,73,179,179,86,259,179,86,179,259,259'
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
        simsLabel.append(label[s[0]])
        
#print simsLabel
counts=Counter(simsLabel)
print counts.most_common()
'''