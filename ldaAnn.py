import logging, string, sys, glob, shutil
from gensim import corpora, models, similarities
from collections import Counter
from random import random
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


ann=[line.strip().split(',') for line in open('annotations.txt')]
#for testing
a2=open('annotations22.txt', 'w+')
l2=open('label22.txt', 'w+')
labels=[line.strip() for line in open('label.txt')]
filelist=list()
count=0
for file in glob.glob('images1/*.jpg'):
    rd=random()
    if rd>0.75:

        for a in ann[count]:
            a2.write(str(a)+'\t')
        a2.write('\n')
        l2.write(str(labels[count])+'\n')
        filename=file.split('/')[len(file.split('/'))-1]
        shutil.move(file, 'test1/'+filename)
        labels.pop(count)
        ann.pop(count)
        continue
    filelist.append(file)
    count+=1
print count
print '---------------------'
#for training
a1=open('annotations21.txt', 'w+')
for a in ann:
    for e in a:
        a1.write(str(e)+'\t')
    a1.write('\n')

a1.close()
l=open('label21.txt', 'w+')
for a in labels:
    l.write(str(a)+'\n')
l.close()


texts=ann
#label=[line.strip() for line in open('label.txt')]

allTokens=sum(texts,[])
tokensLess=set(word for word in set(allTokens) if allTokens.count(word)<3)
texts=[[word for word in text if word not in tokensLess] for text in texts]
#build dicionary and corpus
dictionary=corpora.Dictionary(texts)
dictionary.save('dictionary_ann.dict')
corpus=[dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('corpus_ann.mm', corpus)

#load dictionary
#dictionary=corpora.Dictionary.load('dictionary.dict')
#load corpus
#corpus=corpora.MmCorpus('corpus.mm')
#build lda model
lda=models.LdaModel(corpus, id2word=dictionary, num_topics=8)
lda.save('model_ann.lda')