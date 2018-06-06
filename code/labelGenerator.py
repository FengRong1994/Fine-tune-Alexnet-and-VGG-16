file1=open("test1.txt")
cnt=0
for line1 in file1.readlines():
    a,b=line1.split("\t")
    #print('train')
    #print(a,b)
    file2=open("label.txt")
    for line2 in file2.readlines():
        c,d=line2.split("\t")
        #print('label1')
        #print(c,d)
        if(a==c):
            #print('label2')
            #print(c,d)
            for i in range(int(b)):
                with open ('val.txt','a') as f:
                    if(i<9):
                        f.write(r'/Users/rongfeng/Downloads/data/'+a+'/'+a+'_000'+str(i+1)+r'.jpg'+' '+d)
                    elif((i>=9) and (i<99)):
                        f.write(r'/Users/rongfeng/Downloads/data/'+a+'/'+a+'_00'+str(i+1)+r'.jpg'+' '+d)
                    elif((i>=99)and (i<999)):
                        f.write(r'/Users/rongfeng/Downloads/data/'+a+'/'+a+'_0'+str(i+1)+r'.jpg'+' '+d)
                    else:
                        f.write(r'/Users/rongfeng/Downloads/data/'+a+'/'+a+'_'+str(i+1)+r'.jpg'+' '+d)
                    
            file2.close()
            break