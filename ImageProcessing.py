import cv2;
import numpy as np

img1=cv2.imread ('Square-circle.png',0)

#########Q 1##########
kernelCircle= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
kernelRect= cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
dilation1 = cv2.dilate(img1,kernelCircle,iterations = 1)
dilation2 = cv2.dilate(img1,kernelRect,iterations = 1)
cv2.imwrite('square-circle-1.png',dilation1)
cv2.imwrite('square-circle-2.png',dilation2)

#########Q 2 ########
img2=cv2.imread ('Cameraman.png',0)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
dilation3 = cv2.dilate(img2,kernel,iterations = 1)
cv2.imwrite('cameraman-denoised.png',dilation3

#########Q 3##########
img3=cv2.imread ('lady.png',0)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
img34= cv2.dilate(img3,kernel,iterations = 1)
dilation4= img34- img3
cv2.imwrite('lady-edge.png',dilation4)

#########Q 4##########
img4=cv2.imread ('circle-square.png',0)
kernelCircle= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
kernelRect= cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
erosion1 = cv2.erode(img4,kernelCircle,iterations = 1)
erosion2 = cv2.erode(img4,kernelRect,iterations = 1)
cv2.imwrite('circle-erode.png',erosion1)
cv2.imwrite('square-erode.png',erosion2)

######### Q5 ##########
img5=cv2.imread ('Circles.png',0)
kernelCircle1= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25))
ret,thres= cv2.threshold(img5,127,255,cv2.THRESH_BINARY_INV)
erosion31 = cv2.erode(thres,kernelCircle1,iterations = 1)
cv2.imwrite('Circleerode.png',erosion31)

######### Q6 ########
kernel6 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
erosion6= cv2.erode(img2,kernel6)
cv2.imwrite('cameraman erode.png',erosion6)

######### Q7 a) ########
img6 = cv2.imread('Circle_and_Lines.png',0)
kernel7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
ON = cv2.morphologyEx(img6, cv2.MORPH_OPEN, kernel7)
cv2.imwrite('circle.png',ON)

L =img6-ON
kernel71 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
erosion7= cv2.erode(L,kernel71)
cv2.imwrite('line.png',erosion7)
####### Q7 b) #############

img7 = cv2.imread('circle.png',0)
thr, img7 = cv2.threshold(img7, 127, 255, cv2.THRESH_BINARY)
out = cv2.connectedComponentsWithStats(img7, 8, cv2.CV_32S)

countlabel = out[0]
label = out[1]
stats = out[2]
cent= out[3]
counter = 0

for i in cent:
    counter += -1
    cv2.putText(img7, str(counter), (int(i[0])-5, int(i[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1, 1)
cv2.imwrite('count_circles.png', img7)
#######
img71 = cv2.imread('line.png',0)
thr, img71 = cv2.threshold(img71, 127, 255, cv2.THRESH_BINARY)
img71 = cv2.morphologyEx(img71,cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
img71 = cv2.morphologyEx(img71,cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
out = cv2.connectedComponentsWithStats(img71, 8, cv2.CV_32S)

countlabel = out[0]
label = out[1]
stats = out[2]
cent = out[3]
counter = 0

for i in cent:
    counter += 1
    cv2.putText(img71, str(counter), (int(i[0])-5, int(i[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1, 1)
cv2.imwrite('count_lines.png', img71)

################## Q8 a) #################
img81 = cv2.imread('Girl.png',0)
r, img81 = cv2.threshold(img81, 127, 255, cv2.THRESH_BINARY_INV)
img82 = cv2.imread('Dog.png',0)
r, img82 = cv2.threshold(img82, 127, 255, cv2.THRESH_BINARY_INV)
img83 = cv2.imread('HotBallon.png',0)
r, img83 = cv2.threshold(img83, 127, 255, cv2.THRESH_BINARY_INV)

def skeletonize(img):
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while (not done):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True
    return skel

cv2.imwrite('Dogskeleton.png', skeletonize(img82))
cv2.imwrite('Girlskeleton.png', skeletonize(img81))
cv2.imwrite('HotBallonskeleton.png', skeletonize(img83))

################## Q8 b) #################
img8b1 = cv2.imread('Dog.png', 0)
thre, img8b1 = cv2.threshold(img8b1, 127, 255, cv2.THRESH_BINARY_INV)
m = cv2.moments(img8b1)
cx = int(m['m10'] / m['m00'])
cy = int(m['m01'] / m['m00'])
img8b1o= cv2.circle(img8b1, (cx, cy), 5, (0,0,255), -1)
cv2.imwrite('Dog_center.png', img8b1o)

img8b2 = cv2.imread('Girl.png', 0)
thre, img8b2= cv2.threshold(img8b2, 127, 255, cv2.THRESH_BINARY_INV)
m = cv2.moments(img8b2)
cx = int(m['m10'] / m['m00'])
cy = int(m['m01'] / m['m00'])
img8b2o= cv2.circle(img8b2, (cx, cy), 5, (0,0,255), -1)
cv2.imwrite('Girl_center.png', img8b2o)

img8b3 = cv2.imread('HotBallon.png', 0)
thre, img8b3 = cv2.threshold(img8b3, 127, 255, cv2.THRESH_BINARY_INV)
m = cv2.moments(img8b3)
cx = int(m['m10'] / m['m00'])
cy = int(m['m01'] / m['m00'])
img8b3o= cv2.circle(img8b3, (cx, cy), 5, (0,0,255), -1)
cv2.imwrite('HotBallon_center.png', img8b3o)

################## Q8 c) #################
img8c1 = cv2.imread('Dog.png', 0)
thre, img8c1 = cv2.threshold(img8c1, 127, 255, cv2.THRESH_BINARY_INV)
contours, h = cv2.findContours(img8c1, 1, 2)
area = 0
for i in contours:
    area += cv2.contourArea(i)
cv2.putText(img8c1, str(area), (20,350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1, 1)
cv2.imwrite('Dog_area.png', img8c1)

img8c2 = cv2.imread('Girl.png', 0)
thre, img8c2 = cv2.threshold(img8c2, 127, 255, cv2.THRESH_BINARY_INV)
contours, h = cv2.findContours(img8c2, 1, 2)
area = 0
for i in contours:
    area += cv2.contourArea(i)
cv2.putText(img8c2, str(area), (20,350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1, 1)
cv2.imwrite('Girl_area.png', img8c2)

img8c3 = cv2.imread('HotBallon.png', 0)
thre, img8c3 = cv2.threshold(img8c3, 127, 255, cv2.THRESH_BINARY_INV)
contours, h = cv2.findContours(img8c3, 1, 2)
area = 0
for i in contours:
    area += cv2.contourArea(i)
cv2.putText(img8c3, str(area), (20,350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1, 1)
cv2.imwrite('HotBallon_area.png', img8c3)

#################### Q9 #####################
ker1=np.array([(1,0,0),(0,1,0),(0,0,0)],np.uint8)
ker2=np.array([(0,1,0),(0,1,0),(0,0,0)],np.uint8)
ker3=np.array([(0,0,1),(0,1,0),(0,0,0)],np.uint8)
ker4=np.array([(0,0,0),(1,1,0),(0,0,0)],np.uint8)
ker5=np.array([(0,0,0),(0,1,1),(0,0,0)],np.uint8)
ker6=np.array([(0,0,0),(0,1,0),(1,0,0)],np.uint8)
ker7=np.array([(0,0,0),(0,1,0),(0,1,0)],np.uint8)
ker8=np.array([(0,0,0),(0,1,0),(0,0,1)],np.uint8)

img91 = cv2.imread('Dogskeleton.png', 0)
i1 = cv2.morphologyEx(img91, cv2.MORPH_HITMISS, ker1)
i2 = cv2.morphologyEx(img91, cv2.MORPH_HITMISS, ker2)
i3 = cv2.morphologyEx(img91, cv2.MORPH_HITMISS, ker3)
i4 = cv2.morphologyEx(img91, cv2.MORPH_HITMISS, ker4)
i5 = cv2.morphologyEx(img91, cv2.MORPH_HITMISS, ker5)
i6 = cv2.morphologyEx(img91, cv2.MORPH_HITMISS, ker6)
i7 = cv2.morphologyEx(img91, cv2.MORPH_HITMISS, ker7)
i8 = cv2.morphologyEx(img91, cv2.MORPH_HITMISS, ker8)
img91o = i1 + i2 + i3+ i4+ i5+ i6+ i7+ i8
img91o = img91 - img91o
cv2.imwrite('Dog_end.png',img91o)


img92 = cv2.imread('Girlskeleton.png', 0)
i1 = cv2.morphologyEx(img92, cv2.MORPH_HITMISS, ker1)
i2 = cv2.morphologyEx(img92, cv2.MORPH_HITMISS, ker2)
i3 = cv2.morphologyEx(img92, cv2.MORPH_HITMISS, ker3)
i4 = cv2.morphologyEx(img92, cv2.MORPH_HITMISS, ker4)
i5 = cv2.morphologyEx(img92, cv2.MORPH_HITMISS, ker5)
i6 = cv2.morphologyEx(img92, cv2.MORPH_HITMISS, ker6)
i7 = cv2.morphologyEx(img92, cv2.MORPH_HITMISS, ker7)
i8 = cv2.morphologyEx(img92, cv2.MORPH_HITMISS, ker8)
img92o = i1 + i2 + i3+ i4+ i5+ i6+ i7+ i8
img92o = img92 - img92o
cv2.imwrite('Girl_end.png', img92o)


img93 = cv2.imread('HotBallonskeleton.png', 0)
i1 = cv2.morphologyEx(img93, cv2.MORPH_HITMISS, ker1)
i2 = cv2.morphologyEx(img93, cv2.MORPH_HITMISS, ker2)
i3 = cv2.morphologyEx(img93, cv2.MORPH_HITMISS, ker3)
i4 = cv2.morphologyEx(img93, cv2.MORPH_HITMISS, ker4)
i5 = cv2.morphologyEx(img93, cv2.MORPH_HITMISS, ker5)
i6 = cv2.morphologyEx(img93, cv2.MORPH_HITMISS, ker6)
i7 = cv2.morphologyEx(img93, cv2.MORPH_HITMISS, ker7)
i8 = cv2.morphologyEx(img93, cv2.MORPH_HITMISS, ker8)
img93o = i1 + i2 + i3+ i4+ i5+ i6+ i7+ i8
img93o = img93 - img93o
cv2.imwrite('Hotballon_end.png', img93o)




########
def jun(img):
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    kernel1 = np.array([(1, 0, 1), (0, 1, 0), (0, 1, 0)])
    kernel2 = np.array([(0, 1, 0), (0, 1, 1), (1, 0, 0)])
    kernel3 = np.array([(0, 0, 1), (1, 1, 0), (0, 0, 1)])
    kernel4 = np.array([(1, 0, 0), (0, 1, 1), (0, 1, 0)])
    kernel5 = np.array([(0, 1, 0), (0, 1, 0), (1, 0, 1)])
    kernel6 = np.array([(0, 0, 1), (1, 1, 0), (0, 1, 0)])
    kernel7 = np.array([(1, 0, 0), (0, 1, 1), (1, 0, 0)])
    kernel8 = np.array([(0, 1, 0), (1, 1, 0), (0, 0, 1)])
    kernel9 = np.array([(1, 0, 0), (0, 1, 0), (1, 0, 1)])
    kernel10 = np.array([(1, 0, 1), (0, 1, 0), (1, 0, 0)])
    kernel11 = np.array([(1, 0, 1), (0, 1, 0), (0, 0, 1)])
    kernel12 = np.array([(0, 0, 1), (0, 1, 0), (1, 0, 1)])
    kernel13 = np.array([(1, 0, 1), (0, 1, 1), (0, 1, 0)])
    i11 = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel1)
    i22 = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel2)
    i33 = cv2.morphologyEx(img, cv2.MORPH_HITMISS,kernel3)
    i44 = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel4)
    i55 = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel5)
    i66 = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel6)
    i77 = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel7)
    i88 = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel8)
    i9 = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel9)
    i10 = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel10)
    i11 = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel11)
    i12= cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel12)
    i13 = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel13)
    imgout = i11 + i22 + i33 + i44 + i55 + i66 + i77 + i88 + i9 + i10 + i11 + i12 + i13
    imgout = cv2.dilate(imgout, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    return imgout

cv2.imwrite('Dog_jun.png', jun(img91))
cv2.imwrite('Girl_jun.png', jun(img92))
cv2.imwrite('Hotballon_jun.png', jun(img93))

############### Q10 ###########
img10 = cv2.imread('words.png', 0)
thres, img10 = cv2.threshold(img10, 159, 255, cv2.THRESH_BINARY_INV)
out = cv2.connectedComponentsWithStats(img10, 4, cv2.CV_32S)
num_labels = out[0]
labels = out[1]
stats = out[2]
cent = out[3]
counter = 0
for i in cent:
    counter += 1
cv2.putText(img10, str(num_labels), (350,175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1, 1)
cv2.imwrite('count_words.png', img10)

