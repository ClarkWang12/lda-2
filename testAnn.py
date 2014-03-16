import sys, logging
from gensim import corpora, models, similarities
from collections import Counter
from numpy.random import multinomial
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)




label=[line.strip() for line in open('label21.txt')]
label2=[line.strip() for line in open('label22.txt')]
annotations=[line.strip().split(',') for line in open('annotations21.txt')]
#load dictionary
dictionary=corpora.Dictionary.load('dictionary_ann.dict')
#load corpus
corpus=corpora.MmCorpus('corpus_ann.mm')
#build lda model
#lda=models.LdaModel(corpus, id2word=dictionary, num_topics=100)
#lda.save('model.lda')
lda=models.LdaModel.load('model_ann.lda')


count=0
result=open('resultAnn.txt', 'w+')
for line in open('annotations22.txt'):
    test=line.strip().split()
    print test
    #convert test document
    bow=dictionary.doc2bow(test)
    vec=lda[bow]
    index=similarities.MatrixSimilarity(lda[corpus])
    #calculate similarities
    sims=index[vec]
    #sims is (document_number, document_similarity) 2-tuples
    #sort based on similarities

    sims=sorted(enumerate(sims), key=lambda item: -item[1])


    simsLabel=list()
    for s in sims:
        simsLabel.append(label[s[0]])
        if len(simsLabel)>=20:
            break
        
    #print simsLabel
    counts=Counter(simsLabel)
    topic=counts.most_common(1)[0][0]
    num=counts.most_common(1)[0][1]
    c=label2[count]
    result.write(c+'\t'+str(topic)+'\n')
    print(c, topic)
    count+=1