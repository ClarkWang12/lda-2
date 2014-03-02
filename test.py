import cv2, numpy, sys, logging
from gensim import corpora, models, similarities
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
if len(sys.argv)!=2:
    print 'Need input image'
    sys.exit(1)
img=cv2.imread(sys.argv[1], 0)
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
         
map(str, test)
#TODO: change type of test to string
#begin testing   
############################
''''
info=open('label.txt')
label=list()
for line in info:
    label.append(line.strip())


#load dictionary
dictionary=corpora.Dictionary.load('dictionary.dict')
#load corpus
corpus=corpora.MmCorpus('corpus.mm')
#build lda model
lda=models.LdaModel(corpus, id2word=dictionary, num_topics=100)

#convert test document
bow=dictionary.doc2bow(test)
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