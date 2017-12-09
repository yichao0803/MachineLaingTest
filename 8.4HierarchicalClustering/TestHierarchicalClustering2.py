# -*- coding: utf-8 -*-
"""
@time: 2017/12/9 12:47
@author: Zhang Yichao
@file: TestHierarchicalClustering2.py
"""

from PIL import Image # Notice the 'from PIL' at the start of the line
from PIL import ImageDraw
#from HierarchicalClustering import *
from HierarchicalClustering import hcluster
from HierarchicalClustering import getheight
from HierarchicalClustering import getdepth
from HierarchicalClustering import extract_clusters
from HierarchicalClustering import get_cluster_elements
from HierarchicalClustering import printclust
import numpy as np
import os



def drawdendrogram(clust, imlist, jpeg='cluster.png'):
    # height and width
    h = getheight(clust) * 20
    w = 1200
    depth = getdepth(clust)

    # width is fixed ,so scale distances accardingly
    scaling = float(w - 150) / depth

    # Create a new image with a white background
    img = Image.new("RGB", (w, h), "#fff")
    draw = ImageDraw.Draw(img)

    draw.line((0, h / 2, 10, h / 2), fill=(255, 0, 0))

    # Draw the first node
    drawnode(draw, clust, 10, int(h / 2), scaling, imlist, img)
    img.save(jpeg)


idxImage=1

def drawnode(draw, clust, x, y, scaling, imlist, img):
    if clust.id < 0:
        h1 = getheight(clust.left) * 20
        h2 = getheight(clust.right) * 20
        top = y - (h1 + h2) / 2
        bottom = y + (h1 + h2) / 2
        # Line lenth
        ll = clust.distance * scaling
        # Vertiacl line from this cluster to children
        draw.line((x, top + h1 / 2, x, bottom - h2 / 2), fill=(255, 0, 0))

        # Horizontal line to left item
        draw.line((x, top + h1 / 2, x + ll, top +h1 / 2), fill=(255, 0, 0))

        # Horizontal line to right item
        draw.line((x, bottom - h2 / 2, x + ll, bottom - h2 / 2), fill=(255, 0, 0))

        # Call the function to draw the left and right nodes
        drawnode(draw, clust.left, x + ll, top + h1 / 2, scaling, imlist, img)
        drawnode(draw, clust.right, x + ll, bottom - h2 / 2, scaling, imlist, img)
    else:
        # if this is an endpoint,draw a thumbnail is image
        nodeim = Image.open(imlist[clust.id])
        nodeim.thumbnail((20, 20))
        ns = nodeim.size
        print(x, y - ns[1] // 2)
        print(x + ns[0])
        print(img.paste(nodeim, (int(x), int(y - ns[1] // 2), int(x + ns[0]), int(y + ns[1] - ns[1] // 2))))

        nodeim1 = Image.open(imlist[clust.id])
        #nodeim.thumbnail((20, 20))

        #img = Image.new("RGB", (w, h), "#fff")
        #draw = ImageDraw.Draw(img)
        global idxImage
        nodeim1.save(str(idxImage)+".jpg")
        idxImage+=1



# create a list of image
imlist = []
folderPath = r'..\sunset\flickr'
for filename in os.listdir(folderPath):
    if os.path.splitext(filename)[1] == '.jpg':
        imlist.append(os.path.join(folderPath, filename))
n = len(imlist)
#print(n)

# extract feature vector for each image
features = np.zeros((n, 3))
for i in range(n):
    im = np.array(Image.open(imlist[i]))
    R = np.mean(im[:, :,0].flatten())
    G = np.mean(im[:, :,1].flatten())
    B = np.mean(im[:, :,2].flatten())
    features[i]=np.array([R,G,B])

#print("features")
#print(features)

tree= hcluster(features)
print("get_cluster_elements:")
print(get_cluster_elements(tree))
drawdendrogram(tree,imlist,jpeg='sunset.png')

