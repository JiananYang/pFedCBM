

import os

from glob import glob
cnt = 0

lis = os.listdir("broden")

for i in lis:
    for folder in ['positives','negatives']:
        print (os.path.join("broden",i,folder))
        print (glob(os.path.join("broden",i,folder)))
        cnt += len(glob(os.path.join("broden",i,folder,'*.png')))
        
print (cnt)