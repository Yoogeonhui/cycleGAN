import os
import numpy as np
import cv2


class LoadBatch:
    def __init__(self, directory, step_num=0, batch_size=100, imageW=128, imageH=128):
        self.step_num = step_num
        self.batch_size = batch_size
        self.data = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.png') or file.endswith('.jpg'):
                    self.data.append(os.path.join(root,file))

        if batch_size>len(self.data):
            print('Batch is bigger than Data, this program will set batch_size = data_size data: ', len(self.data))
            self.batch_size = len(self.data)
        self.data_size = len(self.data)
        self.imageW = imageW
        self.imageH = imageH


    def getStep(self):
        return self.step_num

    def getEpoch(self):
        return self.step_num//self.data_size

    def getBatch(self):
        startIdx = self.step_num%self.data_size
        endIdx = (self.step_num+self.batch_size-1)%self.data_size+1

        result = np.zeros([self.batch_size,self.imageH, self.imageW, 3], np.float32)
        if startIdx>endIdx:
            filenames = self.data[startIdx:]+self.data[:endIdx]
        else:
            filenames = self.data[startIdx:endIdx]

        for i, loc in enumerate(filenames):
            im = cv2.imread(loc)
            print(np.shape(im))
            resized = cv2.resize(im,[self.imageH, self.imageW, 3])
            result[i, :, :, :] = resized

        self.step_num += self.batch_size
        return result


