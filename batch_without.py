import cv2, numpy, sys, logging, glob
from gensim import corpora, models, similarities
from collections import Counter
from numpy.random import multinomial
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
result=open('batch_without.txt', 'w+')


label=[line.strip() for line in open('label1.txt')]
#annotations=[line.strip().split(',') for line in open('annotations1.txt')]

#load dictionary
dictionary=corpora.Dictionary.load('dictionary_without.dict')
#load corpus
corpus=corpora.MmCorpus('corpus_without.mm')
#build lda model
#lda=models.LdaModel(corpus, id2word=dictionary, num_topics=100)
#lda.save('model.lda')
lda=models.LdaModel.load('model_without.lda')

for file in glob.glob('test/*.jpg'):
    
    print file
    img=cv2.imread(file, 0)
    sift=cv2.SIFT()
    kp=sift.detect(img, None)

    cfile=open('centroid.txt')
    centroid=list()
    for line in cfile:
        centroid.append(line.strip().split())
    centroid=numpy.array(centroid).astype(numpy.float32)

    test=list()
    for k in kp:
        x, y=k.pt
        x, y=int(x), int(y)
        patch=numpy.array(img[x:x+5, y:y+5]).astype(int)
        n, m=patch.shape

        if n*m==25:
            patch=patch.reshape(1, n*m)
            patch=numpy.array(patch)
            distance=list()
            for c in centroid:
                c=c.reshape(1, 25)
                distance.append(numpy.linalg.norm(c-patch))
            test.append(distance.index(min(distance)))
         
    test=map(str, test)

    #begin testing   
    ############################

    
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
        if s[1]>0.5:
            simsLabel.append(label[s[0]])
        if len(simsLabel)>=20:
            break
        
    #print simsLabel
    counts=Counter(simsLabel)
    topic=counts.most_common(1)[0][0]
    num=counts.most_common(1)[0][1]
    c=file.split('/')[len(file.split('/'))-1]
    c=c.split('.')[0]
    c=c.split('_')[0]
    result.write(c+'\t'+str(topic)+'/n')
    print(c, topic)

result.close()
    
    
    
''''
anns=list()
sz=0
for s in sims:
    if s[1]<0.5:
        break
    elif label[s[0]]==topic:
        for a in annotations[s[0]]:
            anns.append(a)
            sz+=len(a)
sz/=len(anns)
l=float(len(anns))
dist=multinomial(sz, [1/l]*int(l))
a=[anns[i]*dist[i] for i in range(len(anns)) if dist[i]!=0]
print a
'''

#image classification
#text and sift; accuracy
#without lda comparison 
#measure; baseline
#kp+annotation
