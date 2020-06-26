import numpy as np
from  skimage import *
import cv2
img=cv2.imread("test.jpg",0)
print("The shape of img",img.shape)
row,col=img.shape

print("Row:",row)
print("Col",col)
#print("Chann",chann)

cv2.imshow("img",img)
cv2.waitKey(0)
blur_sample=[]
#for i in range(10):

#    temp_img=cv2.GaussianBlur(img,(3,3),i**2,i**2)
    #print("The blur image ",i," is ",cv2.GaussianBlur(img,(i+2,i+2),0))

#    blur_sample.append(temp_img)
    # cv2.imshow(f'Gaussian[{i}]',cv2.GaussianBlur(img,(i+2,i+2),0))
    # cv2.waitKey(0)

#print(len(blur_sample))

#for j in range(10):
     #cv2.imshow(f"Gaussian Blur {j}",blur_sample[j])
     #cv2.waitKey(0)


#print(blur_sample[1]-blur_sample[2])
#print("**************************")
k=3.7
for i in range(10):

    temp_img=cv2.GaussianBlur(img,(5,5),i*2,i*2)
    #print("The blur image ",i," is ",cv2.GaussianBlur(img,(i+2,i+2),0))

    blur_sample.append(temp_img)
    # cv2.imshow(f'Gaussian[{i}].jpg',cv2.GaussianBlur(img,(i+2,i+2),0))
    # cv2.waitKey(0)
for j in range(10):
#    cv2.imshow(f"sigma[{k**j}]",blur_sample[j])
 #   cv2.waitKey(0)
    cv2.imwrite(f'sigma[{k**j}].jpg',blur_sample[j])

#for n in range(7):
    # difference2=blur_sample[n+1]-blur_sample[n+2]


#blur_sample_copy=blur_sample[1].copy()
#print("value is :")
#print(blur_sample_copy-blur_sample[1])
DOG_sample=[]
for m in range(8):
    difference1=blur_sample[m+1]-blur_sample[m]
    #difference2=blur_sample[m+1]-blur_sample[m+2]
    DOG_sample.append(difference1)
    #DOG_sample.append(difference2)
    cv2.imwrite(f"dog{m}.jpg",DOG_sample[m])
    #cv2.imshow(f"diff{m+1}",dic_sample[m+1])
    #cv2.waitKey(0)
    #dic_sample[m+1]=difference2
print(len(DOG_sample))

#for p in range(len(DOG_sample)):
#    cv2.imshow(f"dog{p}",DOG_sample[p])
#    cv2.waitKey(0)
#neighbours=[]
points_set=[]
for count in range(len(DOG_sample)-1):
    img_temp=DOG_sample[count]
    #neighbours=[]
    #if count<len(DOG_sample)-1:
    if count>0:
       #neighbours=[]
       for x in range(img_temp.shape[0]-1):
           for y in range(img_temp.shape[1]-1):
               if x>0 and y>0:
                  neighbours=[]
                  neighbours=[img_temp[x-1,y-1],img_temp[x-1,y],img_temp[x-1,y+1],img_temp[x,y-1],img_temp[x,y+1],img_temp[x+1,y-1],img_temp[x+1,y],img_temp[x+1,y+1]]
                  prev_img=DOG_sample[count-1]
                  neighbours=neighbours+[prev_img[x-1,y-1],prev_img[x-1,y],prev_img[x-1,y+1],prev_img[x,y-1],prev_img[x,y],prev_img[x,y+1],prev_img[x+1,y-1],prev_img[x+1,y],prev_img[x+1,y+1]]
                  next_img=DOG_sample[count+1]
                  neighbours=neighbours+[next_img[x-1,y-1],next_img[x-1,y],next_img[x-1,y+1],next_img[x,y-1],next_img[x,y],next_img[x,y+1],next_img[x+1,y-1],next_img[x+1,y],next_img[x+1,y+1]]
                  if img[x,y]>=max(neighbours):
                     points_set+=[[0,count,x,y]]
                  elif img[x,y]<=min(neighbours):
                       points_set+=[[0,count,x,y]]
    print(f"the key points of img{count}",len(points_set))

print("The total number of key points is",len(points_set))
#points_set2=list(set(points_set))
#print("The total unique of key points is")
print(points_set[800])
print(points_set)






#print(f"The length of img{count}",len(neighbours)," counter is :",count)




#print(len(dic_sample))
#print(dic_sample.values())
################################################
#print(dic_sample.get(1).shape)
#print(list(dic_sample.get(1))[1])
#rint(list(dic_sample.get(1))[2])
#print(list(dic_sample.get(1))[3])
#print(list(dic_sample.get(1))[8])
#print(list(dic_sample.get(6))[2])
#print("************************************")
#print(dic_sample.get(1))

#print("-------------------------------------")

#print(dic_sample.get(3))
#derivative=cv2.GaussianBlur(img,(3,3),5,0)
#cv2.imshow("Gaussian Blur",derivative)
#cv2.waitKey(0)


#difference=blur_sample[1]-blur_sample[4]

#cv2.imshow("Gaussian difference",difference)
#cv2.waitKey(0)

#cv2.imwrite("Diff.jpg",difference)


#cv2.imshow("differ01",blur_sample[1]-blur_sample[0])
#cv2.waitKey(0)
#cv2.imwrite("difference01.jpg",blur_sample[1]-blur_sample[0])

#cv2.imshow("difference21",blur_sample[1]-blur_sample[2])
#cv2.waitKey(0)
#cv2.imwrite("difference21.jpg",blur_sample[1]-blur_sample[2])