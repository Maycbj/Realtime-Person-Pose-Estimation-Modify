import cv2 as cv
import numpy as np
import scipy
import PIL.Image
import math
import caffe
import time
from config_reader import config_reader
import util
import copy
import sys
import matplotlib
import pylab as plt

param, model = config_reader()
#multiplier = [ x * model['boxsize'] / oriImg.shape[0] for x in param['scale_search'] ]
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
           [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
           [1, 16], [16, 18], [3, 17], [6, 18]]
mapIdx = [[31, 32], [39, 40], [33, 34], [35, 36], [41, 42], [43, 44], [19, 20], [21, 22],
          [23, 24], [25, 26], [27, 28], [29, 30], [47, 48], [49, 50], [53, 54], [51, 52],
          [55, 56], [37, 38], [45, 46]]
if param['use_gpu']:
   caffe.set_mode_gpu()
   caffe.set_device(param['GPUdeviceNumber'])
else:
   caffe.set_mode_cpu()

if(len(sys.argv) < 2):
    print 'Please input modelType!'
    sys.exit()
net = caffe.Net(model['deployFile_'+sys.argv[1]], model['caffemodel_'+sys.argv[1]], caffe.TEST)

def resize_from_short_side(input_image_file):
    fixed_size = 224
    oriImg = cv.imread(input_image_file)
    oriImg_x, oriImg_y = oriImg.shape[0], oriImg.shape[1]
#    if oriImg.shape[1] <= oriImg.shape[0] and oriImg.shape[1] > fixed_min_size:
#        oriImg = cv.resize(oriImg, (fixed_min_size, oriImg.shape[0]*fixed_min_size/oriImg.shape[1]), interpolation=cv.INTER_CUBIC)
#    elif oriImg.shape[0] < oriImg.shape[1] and oriImg.shape[0] > fixed_min_size:
#        oriImg = cv.resize(oriImg, (oriImg.shape[1]*fixed_min_size/oriImg.shape[0], fixed_min_size), interpolation=cv.INTER_CUBIC)
    testImg = cv.resize(oriImg, (fixed_size,fixed_size), interpolation=cv.INTER_LINEAR)  
    print('resized (%d, %d) -> (%d, %d)' %(oriImg.shape[0], oriImg.shape[1], testImg.shape[0], testImg.shape[1]))
    return oriImg, testImg


def predict_image_file(input_image_file):
    start_time = time.time()
    oriImg, testImg = resize_from_short_side(input_image_file)
    print 'step0: Read pic took %.2f ms.' % (1000 * (time.time() - start_time))
    start_time = time.time()
    #VGG
    if(sys.argv[1] == 'raw' or sys.argv[1] == 'v' or sys.argv[1] == 'vr'):
        testImg = np.transpose(np.float32(testImg[:, :, :, np.newaxis]), (3,2,0,1)) / 256 - 0.5
        
    #MobileNet
    elif(sys.argv[1] == 'm'):
        testImg = testImg-[103.94, 116.78, 123.68]     
        testImg = np.transpose(np.float32(testImg[:, :, :, np.newaxis]), (3,2,0,1))

    #ResNet
    elif(sys.argv[1] == 'r'):
        #sub
        testImg = testImg - [103.0626238 , 115.90288257 , 123.15163084][::-1]

        #mean
        # MEAN_NPY_PATH = 'model/ResNet_mean.npy'  
        # mean = np.load(MEAN_NPY_PATH)
        # print imageToTest_padded.shape, mean.transpose(1,2,0).shape
        # imageToTest_padded = imageToTest_padded - mean.transpose(1,2,0)

        testImg = np.transpose(np.float32(testImg[:, :, :, np.newaxis]), (3,2,0,1))
    #SqueNet
    elif(sys.argv[1] == 's'):
        #sub
        testImg = testImg - [103.0626238 , 115.90288257 , 123.15163084]     
        testImg = np.transpose(np.float32(testImg[:, :, :, np.newaxis]), (3,2,0,1))
    else:
        print 'Please input modelType!'
        sys.exit()  
    net.blobs['data'].reshape(*(1, 3, testImg.shape[2], testImg.shape[3]))
    net.blobs['data'].data[...] = testImg    
    
    #print 'shape: %s, min: %f, max: %f' %(testImg.shape, testImg.min(), testImg.max())     
    print 'step1: Before CNN took %.2f ms.' % (1000 * (time.time() - start_time))
    start_time = time.time()
    output_blobs = net.forward()
    print 'step2: CNN took %.2f ms.' % (1000 * (time.time() - start_time))
    start_time = time.time()
    heatmap = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[1]].data), (1,2,0))
    paf = np.transpose(np.squeeze(net.blobs[output_blobs.keys()[0]].data), (1,2,0))
    print 'step3: Heatmap and paf process took %.2f ms.' % (1000 * (time.time() - start_time))
    start_time = time.time()
    import scipy
    from scipy.ndimage.filters import gaussian_filter
    all_peaks = []
    peak_counter = 0
    for part in range(18):
        x_list = []
        y_list = []
        map_ori = heatmap[:, :, part]
        starts = time.time() 
        map = map_ori
        #map = gaussian_filter(map_ori, sigma=0.5) 
        map_left = np.zeros(map.shape)
        map_left[1:, :] = map[:-1, :]
        map_right = np.zeros(map.shape)
        map_right[:-1, :] = map[1:, :]
        map_up = np.zeros(map.shape)
        map_up[:, 1:] = map[:, :-1]
        map_down = np.zeros(map.shape)
        map_down[:, :-1] = map[:, 1:]
        peaks_binary = np.logical_and.reduce((map >= map_left, map >= map_right, map >= map_up, map >= map_down, map > param['thre1']))
        peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])
        peaks_with_score = [ x + (map_ori[x[1], x[0]],) for x in peaks ]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [ peaks_with_score[i] + (id[i],) for i in range(len(id)) ]
        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    print 'step4: NMS took %.2f ms.' % (1000 * (time.time() - start_time))
    start_time = time.time()
    connection_all = []
    special_k = []
    mid_num = 20
    for k in range(len(mapIdx)):
        score_mid = paf[:, :, [ x - 19 for x in mapIdx[k] ]]
        candA = all_peaks[limbSeq[k][0] - 1]
        candB = all_peaks[limbSeq[k][1] - 1]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if nA != 0 and nB != 0:
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
                    if norm == 0:
                        norm = 0.001
                    vec = np.divide(vec, norm)
                    startend = zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), np.linspace(candA[i][1], candB[j][1], num=mid_num))
                    vec_x = np.array([ score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] for I in range(len(startend))])
                    vec_y = np.array([ score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] for I in range(len(startend))])
                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts) / len(score_midpts) + min(0.5 * oriImg.shape[0] / norm - 1, 0)
                    criterion1 = len(np.nonzero(score_midpts > param['thre2'])[0]) > 0.8 * len(score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior + candA[i][2] + candB[j][2]])
            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if i not in connection[:, 3] and j not in connection[:, 4]:
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if len(connection) >= min(nA, nB):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])
    print 'step5: Part association took %.2f ms.' % (1000 * (time.time() - start_time))
    start_time = time.time()
    subset = -1 * np.ones((0, 20))
    candidate = np.array([ item for sublist in all_peaks for item in sublist ])
    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:, 0]
            partBs = connection_all[k][:, 1]
            indexA, indexB = np.array(limbSeq[k]) - 1
            for i in range(len(connection_all[k])):
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if subset[j][indexB] != partBs[i]:
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2:
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:
                        subset[j1][:-2] += subset[j2][:-2] + 1
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1

                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif not found and k < 17:
                    row = -1 * np.ones(20)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i, :2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])
    print 'step6: Candidate process took %.2f ms.' % (1000 * (time.time() - start_time))
    start_time = time.time()
    deleteIdx = []
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2] / subset[i][-1] < 0.4:
            deleteIdx.append(i) 

    subset = np.delete(subset, deleteIdx, axis=0)
    cmap = matplotlib.cm.get_cmap('hsv')
    canvas = oriImg.copy()
    scaling = np.asarray([1.0*canvas.shape[i] / heatmap.shape[i] for i in range(2)])[::-1]
    scaling_m = np.mean(scaling)
    for i in range(18):
        rgba = np.array(cmap(1 - i / 18.0 - 1.0 / 36))
        rgba[0:3] *= 255
        for j in range(len(all_peaks[i])): 
            cv.circle(canvas, tuple(int(round(pos)) for pos in ((np.asarray(all_peaks[i][j][0:2]))*scaling)), 2, colors[i], thickness=-1)
#    to_plot = cv.addWeighted(testImg, 0.3, oriImg, 0.7, 0)
    stickwidth = 4
    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            #cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = (((X[0] - X[1])*scaling[1]) ** 2 + ((Y[0] - Y[1])*scaling[0]) ** 2) ** 0.5
            #length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5 * scaling_m
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv.ellipse2Poly((int(round(mY*scaling[0])), int(round(mX*scaling[1]))), (int(length / 2), int(stickwidth)), int(angle), 0, 360, 1)
            cv.fillConvexPoly(canvas, polygon, colors[i])
            #canvas = cv.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    print 'step7: Ouput result took %.2f ms.' % (1000 * (time.time() - start_time))
    start_time = time.time()
    import os
    (picname,extension) = os.path.splitext(os.path.basename(input_image_file))
    cv.imwrite('output/'+ picname + '_res'+ extension, canvas)
    print 'step8: Write picture took %.2f ms.' % (1000 * (time.time() - start_time))

    
#dry_CNN
predict_image_file('input/one_people.jpg')
print '\n\n'

start_time_all = time.time()
import os
file_list = os.listdir('input')
file_list = [file_name for file_name in file_list if file_name.endswith('.jpg')] 
print file_list
for file_name in file_list: 
    start_time_one = time.time()
    predict_image_file('input/' + file_name)
    print file_name + ' took %.2f ms.\n' % (1000 * (time.time() - start_time_one))

print 'Avg process took %.2f ms.' % (1000 * (time.time() - start_time_all) / len(file_list))