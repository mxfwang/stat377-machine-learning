'''
Viola-Jones Face Detection
Madison Wang
STAT37710/CMSC35300
Spring, 2017
'''


import os 
import numpy as np
from math import log
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

size_face = 2000
size_back = 2000
size_patch = 64

class TwoRectangles():
    '''
    The feature looks like:
    (x1, y1)______________________(x1, y2)
        |                             |
        |                             |
        |         DARK AREA           |
        |                             |
        |                             |
    (x2, y1)______________________(x2, y2)
        |                             |
        |                             |
        |         WHITE AREA          |
        |                             |
        |                             |
    (x3, y1)______________________(x3, y2)
    '''
    def __init__(self, x1, y1, x2, y2, x3):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.x3 = x3
    
    def compute_feature(self, iimg):
        dark = iimg[self.x2][self.y2] + iimg[self.x1][self.y1] - iimg[self.x1][self.y2] - iimg[self.x2][self.y1]
        white = iimg[self.x3][self.y2] + iimg[self.x2][self.y1] - iimg[self.x2][self.y2] - iimg[self.x3][self.y1]
        rv = (dark - white) / ((self.x2 - self.x1) * (self.y2 - self.y1))
        return rv


class ThreeRectangles():
    '''
    The feature looks like:
    (x1, y1)__________(x1, y2)________(x1, y3)__________(x1, y4)
        |                 |               |                 |
        |      DARK       |     WHITE     |      DARK       |
        |      AREA       |     AREA      |      AREA       |
        |                 |               |                 |
    (x2, y1)__________(x2, y2)________(x2, y3)__________(x2, y4)
    '''
    def __init__(self, x1, x2, y1, y2, y3, y4):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.y3 = y3
        self.y4 = y4
        
    def compute_feature(self, iimg):
        dark1 = iimg[self.x2][self.y2] + iimg[self.x1][self.y1] - iimg[self.x1][self.y2] - iimg[self.x2][self.y1]
        dark2 = iimg[self.x2][self.y4] + iimg[self.x1][self.y3] - iimg[self.x1][self.y4] - iimg[self.x2][self.y3]
        white = iimg[self.x2][self.y3] + iimg[self.x1][self.y2] - iimg[self.x1][self.y3] - iimg[self.x2][self.y2]
        dark_avg = (dark1 + dark2) / ((self.y2 - self.y1)  * 2)
        white_avg = white / (self.y3 - self.y2)
        return (dark_avg - white_avg) / (self.x2 - self.x1)

    
class Classifier():
    def __init__(self, features, alphas, polarities, thetas):
        self.features = features      #list of features
        self.alphas = alphas          #np array of alphas (weights for features)
        self.polarities = polarities  #np array of polarities from {-1, 1}
        self.thetas = thetas          #np array of thetas (threasholds for each feature)
    def predict(self, iimg):
        vals = np.array([ftr.compute_feature(iimg) for ftr in self.features])
        vals = np.multiply(vals - self.thetas, self.polarities)
        predictions = 2 * (vals >= 0) - 1
        weighted_sum = np.dot(self.alphas, predictions)
        return 2 * ((weighted_sum + 1) >= 0) - 1
        
        
'''
Return the best weak learner with the lowest error rate
as well as the associated polarity in {-1, 1} and threashold theta
Params
-pool: pool of weak learners to draw from
-iimages: integral images of images on which to evaluate the features
-fsize: number of faces in the iimages
-bsize: number of backgrounds in the iimages
Return
-ftr: the best weak learner/feature
-err: error rate over iimages
-p: polarity in {-1, 1}
-theta: threashold
'''
def best_learner(pool, d, iimages, fsize, bsize):
    best_feature = pool[0]
    p = 0
    theta = 0
    err = 1
    total_size = fsize + bsize
    cutoff = 0
    for ftr in pool:
        vals = np.array([ftr.compute_feature(iimg) for iimg in iimages])
        perm = np.argsort(vals)
        
        min_j = 0
        tmp_p = 0
        min_err = 1
        posp_err = np.sum(d[fsize:]) #S^+ + (T^- - S^-) if j == -1
        negp_err = np.sum(d[:fsize]) #S^- + (T^+ - S^+) if j == -1
        for j in range(total_size):
            weight = d[perm[j]]
            if perm[j] < fsize:
                posp_err += weight
                negp_err -= weight
            else:
                posp_err -= weight
                negp_err += weight
            if posp_err < negp_err:
                tp = 1
                tmp_err = posp_err
            else:
                tp = -1
                tmp_err = negp_err
            if tmp_err < min_err:
                min_err = tmp_err
                min_j = j
                tmp_p = tp
        if min_err < err:
            best_feature = ftr
            p = tmp_p
            cutoff = min_j
            if min_j < (total_size - 1):
                theta = 0.5 * (vals[perm[min_j]] + vals[perm[min_j + 1]])
            else:
                theta = vals[perm[min_j]] * 1.1 #Random number
            err = min_err
    vals = np.array([best_feature.compute_feature(iimg) for iimg in iimages])
    perm = np.argsort(vals)
    return (best_feature, err, p, theta)


'''
Build a classifier from a pool of weak learners.
The function runs under strict premise that the the integral images are in the order
[all faces] + [all backgrounds]
Parameters
-pool: pool of weak learners to draw from
-iimages: integral images of images that passed the last classifier in the cascade
-fsize: number of faces in the iimages
-bsize: number of backgrounds in the iimages
Return
-classifier: an instance of the classifier class
'''
def classifier_adaboost(pool, iimages, fsize, bsize):
    print("entered classifier_adaboost")
    d = np.array([0.5 / fsize for i in range(fsize)] + [0.5 / bsize for j in range(bsize)])
    features = []
    alphas = []
    polarities = []
    thetas = []
    y = np.array([1 for i in range(fsize)] + [-1 for i in range(bsize)]) #array of labels
    
    #each element i is sum(a_t * h_t(background[i]))
    #used for evaluating false positive rate
    cum_back_predictions = np.full((1, bsize), 0.0)[0]
    cum_face_predictions = np.full((1, fsize), 0.0)[0]

    while True:
        (ftr, err, p, theta) = best_learner(pool, d, iimages, fsize, bsize)
        vals = (np.array([ftr.compute_feature(iimg) for iimg in iimages]) - theta) * p
        vals = vals > 0
        predictions = vals + (vals - 1) #this operation converts boolean [True, False] to [1, -1] 
        if err == 0:
            err = 0.000000001
        alpha = 0.5 * log((1-err) / err)
        features.append(ftr)
        alphas.append(alpha)
        polarities.append(p)
        thetas.append(theta)
        z = 2 * ((err * (1 - err))**0.5)
        #mult[i] is such that D_{t+1}[i] = D_t[i] * mult[i]
        mult = np.exp(-1 * alpha * np.multiply(y, predictions)) / z
        d = np.multiply(d, mult)
        cum_face_predictions += alpha * predictions[:fsize]
        cum_back_predictions += alpha * predictions[fsize:]
        false_neg = np.sum((cum_face_predictions + 1) < 0) #random number 1 for trying out
        false_pos = np.sum((cum_back_predictions + 1) > 0)
        print("false negative is {}, false positive is {}".format(false_neg, false_pos))
        if (false_neg == 0 and false_pos < (bsize * 0.3)):
            break
    print("exiting classifier_adaboost")
    return Classifier(features, np.array(alphas), np.array(polarities), np.array(thetas))

'''
Auxiliary function
Given a trained cascade and a 64 by 64 integral image, make a prediction
'''
def cascade_predict(cascade, iimg):
    for classifier in cascade:
        if classifier.predict(iimg) == -1:
            return -1
    return 1

'''
main function
train a classifier cascade and predict on the test image
'''
def go():
    iifaces = []
    iibackgrounds = []
    for i in range(size_face):
        img_face = np.array(Image.open("faces/face{}.jpg".format(i)).convert('L'))
        img_background = np.array(Image.open("background/{}.jpg".format(i)).convert('L'))
        iiface = img_face.cumsum(axis=0, dtype=np.int32).cumsum(axis=1, dtype=np.int32)
        iibackgrd = img_background.cumsum(axis=0, dtype=np.int32).cumsum(axis=1, dtype=np.int32)
        iifaces.append(iiface)
        iibackgrounds.append(iibackgrd)
    iimages = np.array(iifaces + iibackgrounds)
    iicopy = np.array(iifaces + iibackgrounds)
    
    pool = []
    #Add TwoRectangal features
    #move vertically and horizontally in strides of 4
    for i in range(15): 
        for j in range(15):
            xul = 4 * i
            yul = 4 * j
            max_height = (63 - xul) // 2
            max_width  = 63 - yul
            pool += [TwoRectangles(xul, yul, xul + k, yul + l, xul + 2 * k) for k in range(1, max_height, 4) for l in range(1, max_width, 4)]

    #Add ThreeRectangle features
    #move vertically and horizontally in strides of 4
    for i in range(15): 
        for j in range(15):
            xul = 4 * i
            yul = 4 * j
            max_height = 63 - xul
            max_width = (63 - yul) // 5
            pool += [ThreeRectangles(xul, xul + k, yul, yul + 2 * l, yul + 3 * l, yul + 5 * l) for k in range(1, max_height, 4) for l in range(1, max_width, 4)]


    cascade = []
    numf = size_face
    numb = size_back
    for i in range(10):
        c = classifier_adaboost(pool, iimages, numf, numb)
        cascade.append(c)
        iimages_f = [iimg for iimg in iimages[:numf]]
        iimages_b = [iimg for iimg in iimages[numf:] if c.predict(iimg) == 1]
        tmp_numb = len(iimages_b)
        if tmp_numb < 20 or tmp_numb == numb:
            break
        else:
            numb = tmp_numb
        iimages = np.array(iimages_f + iimages_b)

    test = np.array(Image.open("class.jpg").convert('L'), dtype=np.int32)
    iitest = test.cumsum(axis=0, dtype=np.int32).cumsum(axis=1, dtype=np.int32)
    
    #use an "ultimate" feature to redduce false positives
    #also use it to evaluate ovelapping patches, and choose the one with the most confidence
    d = np.array([0.5 / size_face for i in range(size_face)] + [0.5 / size_back for j in range(size_back)])
    (ultimate, _, p, theta) = best_learner(pool, d, iicopy, size_face, size_back)
    ultimate_compute = ultimate.compute_feature
    
    #ul[i][j] == 1 if a face is detected in the patch with upper left corner at (i, j)
    ul = np.zeros((1216, 1536), np.int8)
    for i in range(1216): #(1280 - 64) / 4
        for j in range(1536): #(1600 - 64) / 4
            if cascade_predict(cascade, iitest[i : i + size_patch, j : j + size_patch]) == 1:
                ul[i][j] = 1
                
    #use the ultimate feature to eliminate false positives
    values = []
    for i in range(1216):
        for j in range(1536):
            if ul[i][j] == 1:
                val = ultimate_compute(iitest[i : i + size_patch, j : j + size_patch])
                if (val - theta) * p > 20:
                    values.append((ultimate_compute(iitest[i : i + size_patch, j : j + size_patch]) * p, i, j))

                    
    #evaluate each detected patch with the ultimate feature
    #mark (i, j) pair in the matrix ul in the order of descending values
    #for (i, j), if an overlapping patch (i', j') has been marked in ul already
    #(i', j') must have higher feature value (confidence)
    #discard (i, j)
    #if no overlapping patch has been marked, mark (i, j)
    ul = np.zeros((1216, 1536), np.int8)  
    values = sorted(values, key=lambda x: x[0], reverse=True)
    #scan the (size_patch - 10) * 2 region surrounding i, j
    #if the sum is not 0, a patch with higher "confidence" has been marked
    for (_, i, j) in values:
        sub = ul[max(0, i - size_patch + 5) : i + size_patch - 5, max(0, j - size_patch + 5) : j + size_patch - 5]
        s = np.sum(np.sum(sub, axis = 0), axis = 0)
        if s == 0:
            ul[i][j] = 1
            
    fig = plt.figure(figsize=(8, 10), dpi=160)
    ax = fig.add_subplot(111)  
    ax.imshow(test)
    upperleft = np.transpose(np.nonzero(ul))
    for (i, j) in upperleft:
        rect = patches.Rectangle((j, i), size_patch, size_patch, linewidth=1,edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.savefig('class_detection.jpg', bbox_inches='tight', dpi=160)
    plt.show()
    plt.close()

    
go()