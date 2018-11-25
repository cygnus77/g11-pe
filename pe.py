import sys
import os
import cv2 as cv
import numpy as np
import rawpy
import exifread

def showImg(img):
    win = cv.namedWindow('image')
    cv.imshow('image',img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def getCentroid(img_path):
    with open(img_path, 'rb') as img_f:
        tags = exifread.process_file(img_f, details=False, stop_tag="DateTimeOriginal")
        image_timestamp = tags["Image DateTimeOriginal"].values

    with rawpy.imread(img_path) as raw:
        img = raw.postprocess()

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # erosion_size = 5
    # erosion_type = cv.MORPH_RECT
    # element = cv.getStructuringElement(erosion_type, (2*erosion_size + 1, 2*erosion_size+1), (erosion_size, erosion_size))
    # img = cv.erode(img, element)

    _,img = cv.threshold(img, 200, 255, cv.THRESH_BINARY)

    # list of pixels that are white
    rows, cols = np.where(img == 255)
    # avrow, avcol = int(np.average(rows)), int(np.average(cols))

    # isolate stars, each blob contains area and bounding box
    mask = np.zeros((img.shape[0]+2, img.shape[1]+2), dtype=np.uint8)
    blobs = []
    for (row,col) in zip(rows,cols):
        if mask[row+1, col+1] == 0:
            xy = (col,row)
            area, _, _, rect = cv.floodFill(img, mask=mask, seedPoint=xy, newVal=255, flags=cv.FLOODFILL_MASK_ONLY)
            blobs.append( (area,rect) )

    # sort by area
    blobs.sort(key=lambda x:x[0], reverse=True)

    # img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    # for (ret,rect) in blobs:
    #     (x,y,w,h) = rect
    #     img[y:y+h, x:x+w, :]=[0,0,255]
    # img = cv.resize(img, dsize=(0,0), fx = 0.05, fy=0.05)
    # showImg(img)

    # return centroid of biggest star
    (x,y,w,h) = blobs[0][1]
    return (image_timestamp, int(x+w/2), int(y+h/2))

DIR='/home/anand/Downloads/Star3'
# images from Stars2/5741 to 5811
# images from Stars3/6038 to 6219
for imgno in range(6038, 6219):
    img_path = '{0}/DSC_{1}.NEF'.format(DIR, imgno)
    image_timestamp, cx, cy = getCentroid(img_path)
    print(imgno, image_timestamp, cx, cy)
