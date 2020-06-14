import numpy as np 
from tqdm import tqdm
import cv2


def import_img(filepath):
    img = cv2.imread(filepath,cv2.IMREAD_COLOR)
    # cv2.imshow('image',img)
    # cv2.waitKey(0)
    return img


def gray_preprocess_img(img):
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(blur)

    gray = gray/maxVal

    # gray = cv2.threshold(gray,0.5,1,cv2.THRESH_BINARY)[1]
    
    # cv2.imshow("gray", cv2.resize(gray,(2550,1440)))
    # cv2.waitKey(0)

    return gray

def find_stars(img):
    img = img*255
    stars_list = []   #2d list [index, [values]]

    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 125
    params.maxThreshold = 255

    params.filterByColor = False

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 12
    params.maxArea = 300

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.3

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.5

    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.5

    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3 :
        blob_detector = cv2.SimpleBlobDetector(params)
    else : 
        blob_detector = cv2.SimpleBlobDetector_create(params)
   
    blob_detector.empty()
    points = blob_detector.detect(img.astype('uint8'))


    # Show blobs
    # im_with_keypoints = cv2.drawKeypoints(img.astype('uint8'), points, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imshow("Keypoints", cv2.resize(im_with_keypoints,(2550,1440)))
    # cv2.waitKey(0)

    for i in range(len(points)):
        if int(points[i].pt[0]) > 10 and int(points[i].pt[0]) < int(img.shape[1]-10) and int(points[i].pt[1]) > 10 and int(points[i].pt[1]) < int(img.shape[0]-10):
            stars_list.append([int(points[i].pt[0]),int(points[i].pt[1]),-1, int(points[i].size/2),-1])

    return stars_list

def find_closest(star_array,img_index,pos, threshold):

    closest_index = -1
    closest_distance = 9999

    array = np.array(star_array[img_index-1])
    dist_array = np.sum((array[:,:2]-pos)**2,axis=1)
    dist_array = np.sqrt(dist_array)
    if dist_array[np.argmin(dist_array)] < threshold and dist_array[np.argmin(dist_array)] > 3:
        closest_index = np.argmin(dist_array)

    return closest_index

def cut(img,x,y,r):
    # cv2.imshow("Keypoints", cv2.resize(img,(2550,1440)))
    # cv2.waitKey(0)

    mask = np.zeros(img.shape,dtype=np.uint8)
    cv2.circle(mask,(x,y),r,(1,1,1),-1,4,0)

    # cv2.imshow("Keypoints", cv2.resize(cut_out,(1920,1080)))
    # cv2.waitKey(0)
    out = (img*mask)[y-r:y+r+1,x-r:x+r+1]

    out[out<125] = 0

    return out

def cut_out(img,x,y,r):  
    mask_0 = np.ones(img.shape,dtype=np.uint8)
    cv2.circle(mask_0,(x,y),r,(0,0,0),-1,4,0)
    img = img*mask_0

    yp = int(y+r+1)
    yn = int(y-r)
    xp = int(x+r+1)
    xn = int(x-r)
    colors = img[yp,xp,:]

    mask_1 = np.zeros(img.shape,dtype=np.uint8)
    cv2.circle(mask_1,(x,y),r,(int(colors[0]),int(colors[1]),int(colors[2])),-1,4,0)
    img = img+mask_1

    # cv2.imshow("Keypoints", cv2.resize(cut_img,(1920,1080)))
    # cv2.waitKey(0)

    return img


def paste(base_img, paste_img, pos):

    x = pos[0]
    y = pos[1]

    r = int((paste_img.shape[0]-1)/2)

    base_img[y-r:y+r+1,x-r:x+r+1] = base_img[y-r:y+r+1,x-r:x+r+1] + paste_img

    return base_img


def main():
    # input photos
    num_photos = 10

    #tweak these values
    alpha = 1
    beta = 2.5  #higher = darker stars
    distance_threshold = 35  #stars should be close together
    binary_threshold = 0.5

    color_imgs = []
    gray_imgs = []
    star_array = []  #[img][index][x/y/previous_index/starsize/index_on_first]

    print("finding stars")
    for i in tqdm(range(num_photos)):  #find all stars loop
        color_imgs.append(import_img('input/'+str(i+1)+'.JPG'))  #import color img
        gray_imgs.append(gray_preprocess_img(color_imgs[i]))   #make gray versions
        cv2.imwrite('output/' + 'gray_'+str(i)+'.jpg', gray_imgs[i]*255)
        star_array.append(find_stars(gray_imgs[i]))  #find stars
        if i > 0:
            for j in range(len(star_array[i])):
                star_array[i][j][2] = find_closest(star_array,i,[star_array[i][j][0],star_array[i][j][1]],distance_threshold)


    for i in range(num_photos): #find the index on the base_image star, if it exists
        if i > 0:
            for j in range(len(star_array[i])):
                index = star_array[i][j][2]
                for z in reversed(range(i)):      
                    if index != -1 and z != 0:
                        index = star_array[z][index][2]
                star_array[i][j][4] = index

    print("stacking stars")
    base_img = color_imgs[0]/(num_photos*alpha)
    for i in range(num_photos):  #cut and paste stars individually to the base image
        if i > 0: #ignore the first image
            # bad = 0
            for j in tqdm(range(len(star_array[i]))): #for each star
                if star_array[i][j][4] != -1:    
                    cut_star = cut(color_imgs[i],star_array[i][j][0],star_array[i][j][1],star_array[i][j][3]+2)
                    color_imgs[i] = cut_out(color_imgs[i],star_array[i][j][0],star_array[i][j][1],star_array[i][j][3]+4)
                    if cut_star.shape[0] == cut_star.shape[1]:
                        base_img = paste(base_img,cut_star/(num_photos*beta),star_array[0][star_array[i][j][4]][0:2])
                # else:
                #     # print("no path to base image star!")
                #     bad += 1
            
            # bad = 100*bad/len(star_array[i])
            # print(str(bad)+'%' + ' of stars not linked')

    print("stacking images")
    for i in tqdm(range(num_photos)): #stack the all the images together
        if i > 0: #ignore the first image
            base_img = base_img + color_imgs[i]/(num_photos*alpha)


    print("done")
    cv2.imwrite('output/' + "result.png", base_img)


if __name__ == '__main__':
    main()