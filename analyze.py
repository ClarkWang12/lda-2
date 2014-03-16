filename='batch.txt'
outside=['mountain', 'opencountry', 'forest']
city=['insidecity', 'street', 'tallbuilding']
dist=dict()
test=list()
for line in open(filename):
    res=line.strip().split()
    ''''
    for idx in range(len(res)):
        if res[idx] in outside:
            res[idx]='outside'
        if res[idx] in city:
            res[idx]='city'
    '''
    if dist.has_key(res[0]):
        if res[0]==res[1]:
            dist[res[0]][0]+=1
        else:
            dist[res[0]][1]+=1
    else:
        if res[0]==res[1]:
            dist[res[0]]=[1, 0]
        else:
            dist[res[0]]=[0, 1]
for key in dist.keys():
    print(key, dist[key])
