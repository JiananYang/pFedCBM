

import os


#write broden concepts

lis = os.listdir('broden')
i = 0
with open("concept_handbook/broden.txt",'w') as f:
	for concept in lis:
		f.writelines("{} {}\n".format(i,concept))
		i += 1
print (lis)