import os
import numpy as np
list1=os.listdir('/home/common/liqi/data/LINEMOD_6D/LM6d_origin/test/01/annots/bbox')
a=len(list1)
for i  in range(0,a):
    read=np.load(os.path.join('/home/common/liqi/data/LINEMOD_6D/LM6d_origin/test/01/annots/bbox','%i.npy' %(i)))
    np.save(os.path.join('/home/common/liqi/data/LINEMOD_6D/LM6d_origin/test/01/annots/bbox','%04i.npy' %(i)),read)