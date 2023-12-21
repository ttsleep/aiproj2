path='F:\Share\dataset\TT100K\VOCdevkit\VOC2007\JPEGImages'
path2='F:\Share\dataset\TT100K\VOCdevkit\VOC2007\Annotations'
import os
new_arr=os.listdir(path)
new_arr1=os.listdir(path2)
print(new_arr)
print(new_arr1)

# for s,i,j in enumerate(os.listdir(path),os.listdir(path2)):
#     print(i,j)
#     exit()