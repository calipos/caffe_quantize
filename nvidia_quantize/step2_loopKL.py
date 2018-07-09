import sys
import os
sys.path.append(sys.path[0]+"/../../python")
#print sys.path
from caffe.model_libs import *

src_prototxt="pelee_nobn.prototxt"
noInPlace_prototxt="./temp/pelee_nobn_noInPlace.prototxt"
check_if_exist(src_prototxt)
make_if_not_exist("./temp")

class LayerNodeInfo():
    def __init__(self,name,bottom,top,lineStart,lineEnd,levelIdx):
        self.name=name
        self.top=top
        self.bottom=bottom
        self.lineStart=lineStart
        self.lineEnd=lineEnd
        self.levelIdx=levelIdx

def getBNconvs(deploy_nobn_prototxt):
    checkBnInPlace=True
    fileObj = open(deploy_nobn_prototxt,'r')
    lines=fileObj.readlines()
    fileObj.close()
    lineStrats=[]
    for idx,line in enumerate(lines):
        if line.strip().replace(' ','').find("layer{")!=-1:lineStrats.append(idx)
    lineStrats.append(len(lines))
    convBNs=[]
    for i in range(len(lineStrats)-3):
        lineStart1=lineStrats[i]
        lineEnd1=lineStrats[i+1]
        lineStart2=lineStrats[i+1]
        lineEnd2=lineStrats[i+2]
        lineStart3=lineStrats[i+2]
        lineEnd3=lineStrats[i+3]
        findConv=False
        findBN=False
        findSacle=False
        convName1=""
        bnName2=""
        scaleName3=""
        BNbottom=""
        BNtop=""
        SCALEbottom=""
        SCALEtop=""
        for j in range(lineStart1,lineEnd1):
            if lines[j].strip().replace(' ','').find("type:\"Convolution\"")!=-1:findConv=True
            if lines[j].strip().replace(' ','').find("name:\"")!=-1:
                convName1 = lines[j].strip().replace(' ','').replace('\"','').split(':')[1]

        for j in range(lineStart2,lineEnd2):
            if lines[j].strip().replace(' ','').find("type:\"BatchNorm\"")!=-1:findBN=True
            if lines[j].strip().replace(' ','').find("name:\"")!=-1:
                bnName2 = lines[j].strip().replace(' ','').replace('\"','').split(':')[1]
            if checkBnInPlace:
                if lines[j].strip().replace(' ','').find("bottom:\"")!=-1:
                    BNbottom = lines[j].strip().replace(' ','').replace('\"','').split(':')[1]
                if lines[j].strip().replace(' ','').find("top:\"")!=-1:
                    BNtop = lines[j].strip().replace(' ','').replace('\"','').split(':')[1]
        for j in range(lineStart3,lineEnd3):
            if lines[j].strip().replace(' ','').find("type:\"Scale\"")!=-1:findSacle=True
            if lines[j].strip().replace(' ','').find("name:\"")!=-1:
                scaleName3 = lines[j].strip().replace(' ','').replace('\"','').split(':')[1]
            if checkBnInPlace:
                if lines[j].strip().replace(' ','').find("bottom:\"")!=-1:
                    SCALEbottom = lines[j].strip().replace(' ','').replace('\"','').split(':')[1]
                if lines[j].strip().replace(' ','').find("top:\"")!=-1:
                    SCALEtop = lines[j].strip().replace(' ','').replace('\"','').split(':')[1]

        if findConv and findBN and findSacle:
            convBNs.append([convName1,bnName2,scaleName3])
            assert BNbottom == BNtop and SCALEbottom == SCALEtop
    return convBNs

def modifyInPlace(deploy_nobn_prototxt,deploy_nobn_noInPlace_prototxt):
    fileObj = open(deploy_nobn_prototxt,'r')
    lines=fileObj.readlines()
    fileObj.close()
    lineStrats=[]
    protoLines=[]
    for idx,line in enumerate(lines):
        protoLines.append(line)
        if line.strip().replace(' ','').find("layer{")!=-1:lineStrats.append(idx)
    lineStrats.append(len(lines))
    noInPlace=True
    while noInPlace:
        InPlaceNames=[]
        for i in range(len(lineStrats)-1):
            InPlaceName=""
            bottoms=[]
            tops=[]
            for j in range(lineStrats[i],lineStrats[i+1]):
                if protoLines[j].strip().replace(' ','').find("bottom:")!=-1:bottoms.append(protoLines[j].strip().replace(' ','').split('\"')[1])
                if protoLines[j].strip().replace(' ','').find("top:")!=-1:tops.append(protoLines[j].strip().replace(' ','').split('\"')[1])
            if len(bottoms)==len(tops) and len(tops)==1 and bottoms[0]==tops[0] : 
                InPlaceNames.append(bottoms[0])
        if len(InPlaceNames)==0 : break
        #print InPlaceNames
        for i in range(len(lineStrats)-1):
            InPlaceName=""
            bottoms=[]
            tops=[]
            for j in range(lineStrats[i],lineStrats[i+1]):
                if protoLines[j].strip().replace(' ','').find("name:")!=-1 : InPlaceName = protoLines[j].strip().replace(' ','').split('\"')[1]
                if protoLines[j].strip().replace(' ','').find("bottom:")!=-1:bottoms.append(protoLines[j].strip().replace(' ','').split('\"')[1])
                if protoLines[j].strip().replace(' ','').find("top:")!=-1:tops.append(protoLines[j].strip().replace(' ','').split('\"')[1])
            for InPlaceName in InPlaceNames:
                if InPlaceName in bottoms and InPlaceName in tops:
                    for j in range(lineStrats[i],lineStrats[i+1]):
                        #if protoLines[j].strip().replace(' ','').find("bottom:\""+InPlaceName+"\"")!=-1 : protoLines[j] = "  bottom: \""+InPlaceName+"_noInP\"\n"
                        if protoLines[j].strip().replace(' ','').find("top:\""+InPlaceName+"\"")!=-1 : protoLines[j] = "  top: \""+InPlaceName+"_noInP\"\n"
                if InPlaceName in bottoms and InPlaceName not in tops:
                    for j in range(lineStrats[i],lineStrats[i+1]):
                        if protoLines[j].strip().replace(' ','').find("bottom:\""+InPlaceName+"\"")!=-1 : protoLines[j] = "  bottom: \""+InPlaceName+"_noInP\"\n"
                        #if protoLines[j].strip().replace(' ','').find("top:\""+InPlaceName+"\"")!=-1 : protoLines[j] = "  top: \""+InPlaceName+"_noInP\"\n"
    fileObj = open(deploy_nobn_noInPlace_prototxt,'w')
    fileObj.writelines(protoLines)
    fileObj.close()

def getLayers(deploy_nobn_prototxt):
    fileObj = open(deploy_nobn_prototxt,'r')
    lines=fileObj.readlines()
    fileObj.close()
    lineStrats=[]
    protoLines=[]
    for idx,line in enumerate(lines):
        protoLines.append(line)
        if line.strip().replace(' ','').find("layer{")!=-1:lineStrats.append(idx)
    lineStrats.append(len(lines))
    nodesInfo={}
    for i in range(len(lineStrats)-1):
        name=""
        bottoms=[]
        tops=[]
        for j in range(lineStrats[i],lineStrats[i+1]):
            if protoLines[j].strip().replace(' ','').find("name:")!=-1 : name = protoLines[j].strip().replace(' ','').split('\"')[1]
            if protoLines[j].strip().replace(' ','').find("bottom:")!=-1:bottoms.append(protoLines[j].strip().replace(' ','').split('\"')[1])
            if protoLines[j].strip().replace(' ','').find("top:")!=-1:tops.append(protoLines[j].strip().replace(' ','').split('\"')[1])
        if len(bottoms) == 1 and bottoms[0]=='data' : 
            nodesInfo[name] = LayerNodeInfo(name,bottoms,tops,lineStrats[i],lineStrats[i+1],1)
            #print nodesInfo[name].name, nodesInfo[name].bottom, nodesInfo[name].top, nodesInfo[name].levelIdx
        else : 
            this_level=-1
            for bottom in bottoms:
                for key in nodesInfo.keys():
                    if bottom in nodesInfo[key].top: 
                        tmp=nodesInfo[key].levelIdx+1 if this_level<nodesInfo[key].levelIdx+1 else this_level
                        this_level=tmp
                    else: continue
            if this_level>0 : 
                nodesInfo[name] = LayerNodeInfo(name,bottoms,tops,lineStrats[i],lineStrats[i+1],this_level)
                #print nodesInfo[name].name, nodesInfo[name].bottom, nodesInfo[name].top, nodesInfo[name].levelIdx
            else : 
                print "some layer has not been found!"
                assert False
    if len(nodesInfo.keys())!=len(lineStrats)-1 : 
        print "some layer has not been found!"
        assert False
    levelIdxs=[]
    for key in nodesInfo.keys():
        if nodesInfo[key].levelIdx not in levelIdxs:
            levelIdxs.append(nodesInfo[key].levelIdx)
    if max(levelIdxs)!= len(levelIdxs):
        print "some layer missing!"
        assert False
    return lineStrats,protoLines,nodesInfo,levelIdxs

def generateNewProto(protoLines,nodesInfo,level):
    fileObj = open("./temp/new.prototxt",'w')
    for i in range(1,level+1):
        for key in nodesInfo.keys():
            if nodesInfo[key].levelIdx==i:
                lineStart=nodesInfo[key].lineStart
                lineEnd=nodesInfo[key].lineEnd
                for j in range(lineStart,lineEnd):
                    fileObj.write(protoLines[j])
                    if i==level and protoLines[j].strip().replace(' ','').find("name:")!=-1 :fileObj.write("  propagate_down: false\n")
    fileObj.close()        
if __name__ == '__main__':
    assert 0 == len(getBNconvs(src_prototxt))
    modifyInPlace(src_prototxt,noInPlace_prototxt)

    lineStrats,protoLines,nodesInfo,levelIdxs = getLayers(noInPlace_prototxt)

    generateNewProto(protoLines,nodesInfo,3)
    exit(0)
    for i in range(min(levelIdxs),max(levelIdxs)+1):
        generateNewProto(protoLines,nodesInfo,i)
    #for key in nodesInfo.keys():print nodesInfo[key].name, nodesInfo[key].bottom, nodesInfo[key].top, nodesInfo[key].levelIdx

    
    #node = LayerNodeInfo("this_name",[1,2,3],[4,5,6],1,10)    print node.name,node.top,node.bottom,node.lineStart,node.lineEnd
    
    #print lineStrats
    #print protoLines