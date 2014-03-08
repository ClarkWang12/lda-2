import glob
from xml.dom.minidom import parse
images=glob.glob('images/*.jpg')
anotations=open('anotations.txt', 'w+')
label=open('label.txt', 'w+')
for img in images:
    name=img.split('/')[1].strip('.jpg')
    print '--------------'+name+'-----------------------'
    ant='annotations/'+name+".xml"
    tree=parse(ant)
    nodes=tree.getElementsByTagName('name')
    text=str()
    for node in nodes:
        for t in node.childNodes:
            if t.nodeType==t.TEXT_NODE:
                text+=t.nodeValue+','
    text+='\n'
    anotations.write(text)
    label.write(name.split('_')[0]+'\n')