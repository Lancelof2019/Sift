import numpy as np

import cv2

img_org1=cv2.imread("France1.jpg")
img_org2=cv2.imread("France2.jpg")
#print("The pixcel of img is")
#print(img_org1.shape)
gray_img1=cv2.cvtColor(img_org1,cv2.COLOR_BGR2GRAY)
gray_img2=cv2.cvtColor(img_org2,cv2.COLOR_BGR2GRAY)
sift1=cv2.xfeatures2d.SIFT_create()

keypoints1,descriptors1=sift1.detectAndCompute(gray_img1,None)
keypoints2,descriptors2=sift1.detectAndCompute(gray_img2,None)
show_img1=cv2.drawKeypoints(gray_img1,keypoints1,img_org1)
show_img2=cv2.drawKeypoints(gray_img2,keypoints1,img_org2)
cv2.imshow("keypoints1",show_img1)

cv2.waitKey(0)

cv2.imwrite("show1.jpg",show_img1)

cv2.imshow("Keypoints2",show_img2)
cv2.waitKey(0)
cv2.imwrite("show2.jpg",show_img2)


#print("key ponits")
#print(keypoints1)
#print(len(keypoints1))
#print("*******************************")
#print("descriptors")
#print(descriptors1)
#print(descriptors1.shape)


###I try to write KNN for distance####

min=[0,0]
#knnSet=np.zeros(((descriptors1.shape[0]), 2))
knn_list=[]
print(descriptors1.shape[0])
print(descriptors2.shape[0])
record_point=[]
for i in range(descriptors1.shape[0]):
    initial_distance=cv2.norm(descriptors1[i],descriptors2[0],cv2.NORM_L2)
    min[0]=initial_distance
    min[1]=initial_distance
    for j in range(descriptors2.shape[0]):
        distance=cv2.norm(descriptors1[i],descriptors2[j],cv2.NORM_L2)
        #if i==0 and j==0:
          # min[0]=distance
           #min[1]=distance
        if  min[0]>distance:
            temp1=min[0]
            # min0_index=j
            min[0]=distance
            min[1]=temp1
            desc2_index=j
            desc1_index=i
            desc3_index=j-1
        elif min[1]>distance:
             min[1]=distance
             desc3_index=j
             desc1_index=i
    # print("******************************************************************************************************")
    # print(f"The shortest distance between descriptor1[{desc1_index}] and descriptor2[{desc2_index}] is min[0]")
    # print(min[0])
    # print(f"The second shortest distance between descriptor1[{desc1_index}] and descriptor2[{desc3_index}] is min[1]")
    # print(min[1])
    # print("******************************************************************************************************")
    #knnSet[i][0]=min[0]
    #knnSet[i][1]=min[1]
    #knnSet[i][0]=cv2.DMatch(i,desc2_index,min[0])
    #knnSet[i][1]=cv2.DMatch(i,desc3_index,min[1])
    knn_list=knn_list+[[cv2.DMatch(i,desc2_index,min[0]),cv2.DMatch(i,desc3_index,min[1])]]
    #print("*******************************************************")
    #print(f"The Dmatch between {i} and {desc2_index} is :")
    #print(cv2.DMatch(i,desc2_index,min[0]))
    #print(f"The Dmatch between {i} and {desc3_index} is :")
    #print(cv2.DMatch(i,desc3_index,min[0]))a
print(len(knn_list))

knnSet=np.array(knn_list)
print(knnSet.shape)
print(knnSet[9][0])
print(knnSet[9][1])
#print(len(knnSet))
#print(knnSet.size)
###the elements
#print(len(knnSet))
[root@linux sift]# vim sift_test.py
[root@linux sift]# vim sift_test.py
[root@linux sift]# vim sift_test.py
[root@linux sift]# cat sift_test.py
import numpy as np

import cv2

img_org1=cv2.imread("France1.jpg")
img_org2=cv2.imread("France2.jpg")
#print("The pixcel of img is")
#print(img_org1.shape)
gray_img1=cv2.cvtColor(img_org1,cv2.COLOR_BGR2GRAY)
gray_img2=cv2.cvtColor(img_org2,cv2.COLOR_BGR2GRAY)
sift1=cv2.xfeatures2d.SIFT_create()

keypoints1,descriptors1=sift1.detectAndCompute(gray_img1,None)
keypoints2,descriptors2=sift1.detectAndCompute(gray_img2,None)
show_img1=cv2.drawKeypoints(gray_img1,keypoints1,img_org1)
show_img2=cv2.drawKeypoints(gray_img2,keypoints1,img_org2)
cv2.imshow("keypoints1",show_img1)

cv2.waitKey(0)

cv2.imwrite("show1.jpg",show_img1)

cv2.imshow("Keypoints2",show_img2)
cv2.waitKey(0)
cv2.imwrite("show2.jpg",show_img2)


#print("key ponits")
#print(keypoints1)
#print(len(keypoints1))
#print("*******************************")
#print("descriptors")
#print(descriptors1)
#print(descriptors1.shape)


###I try to write KNN for distance####

min=[0,0]
#knnSet=np.zeros(((descriptors1.shape[0]), 2))
knn_list=[]
print(descriptors1.shape[0])
print(descriptors2.shape[0])
record_point=[]
for i in range(descriptors1.shape[0]):
    initial_distance=cv2.norm(descriptors1[i],descriptors2[0],cv2.NORM_L2)
    min[0]=initial_distance
    min[1]=initial_distance
    for j in range(descriptors2.shape[0]):
        distance=cv2.norm(descriptors1[i],descriptors2[j],cv2.NORM_L2)
        #if i==0 and j==0:
          # min[0]=distance
           #min[1]=distance
        if  min[0]>distance:
            temp1=min[0]
            # min0_index=j
            min[0]=distance
            min[1]=temp1
            desc2_index=j
            desc1_index=i
            desc3_index=j-1
        elif min[1]>distance:
             min[1]=distance
             desc3_index=j
             desc1_index=i
    # print("******************************************************************************************************")
    # print(f"The shortest distance between descriptor1[{desc1_index}] and descriptor2[{desc2_index}] is min[0]")
    # print(min[0])
    # print(f"The second shortest distance between descriptor1[{desc1_index}] and descriptor2[{desc3_index}] is min[1]")
    # print(min[1])
    # print("******************************************************************************************************")
    #knnSet[i][0]=min[0]
    #knnSet[i][1]=min[1]
    #knnSet[i][0]=cv2.DMatch(i,desc2_index,min[0])
    #knnSet[i][1]=cv2.DMatch(i,desc3_index,min[1])
    knn_list=knn_list+[[cv2.DMatch(i,desc2_index,min[0]),cv2.DMatch(i,desc3_index,min[1])]]
    #print("*******************************************************")
    #print(f"The Dmatch between {i} and {desc2_index} is :")
    #print(cv2.DMatch(i,desc2_index,min[0]))
    #print(f"The Dmatch between {i} and {desc3_index} is :")
    #print(cv2.DMatch(i,desc3_index,min[0]))a
print(len(knn_list))

knnSet=np.array(knn_list)
print(knnSet.shape)
print(knnSet[9][0])
print(knnSet[9][1])
ratio_threshold=0.88
matches=[]
for m,n in knnSet:
    if m.distance < ratio_threshold*n.distance:
       #matches.append([m])a
       print("*****************************")
       print(m)
       print(m.distance)
       print("*****************************")
       matches.append(m)

print(matches)

#print(len(matches))

img_matches=cv2.drawMatches(gray_img1, keypoints1, gray_img2, keypoints2, matches,outImg=None, matchColor=(0, 255, 0), singlePointColor=(0, 255, 0), flags=2)

cv2.imshow("performance",img_matches)
cv2.waitKey(0)
cv2.imwrite("compare_performance.jpg",img_matches)
#print(len(knnSet))
#print(knnSet.size)
###the elements
#print(len(knnSet))
