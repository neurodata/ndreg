#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import math, copy, csv, sys
from numpy import mat, multiply

class landmarks:
    """
    The landmarks class can be used to read, write and manipulate landmarks.  For example...
        lmk = landmarks()                                       # Construct
        lmk = landmarks([ ['label1',1.5,2], ['label2',2.4,5]] ) # Construct from landmarks list
        lmk = landmarks("/path/to/landmarks.lmk")               # Construct from landmarks file
        lmk.Read(path)           # Read landmarks from given path
        labels = lmk.GetLabels() # Get the labels of the landmarks
        lmk.SetLabels(labels)    # Set the labels of the landmarks
        lmk.Write(path)           # Write landmarks to given path
    """

    def __init__(self, inputLandmarks=[['0',0,0,0]], spacing=[1.0,1.0,1.0]):
        self.spacing = spacing
        if type(inputLandmarks) is str:
            self.Read(inputLandmarks, spacing)
        elif type(inputLandmarks) is list:
            self.SetLandmarks(inputLandmarks)
        else:
            raise Exception("Landmarks must be constructed with either a landmaks list or landmarks file path")

    def SetLandmarks(self,landmarkList):
        """
        Sets the landmarks using given landmark list.
        Format of landmarkList must be...
            [[label1,x1,y1,...],...,[labelN,xN,yN,...]]
        ... where x1,y1,...,xN,yN... must be numeric.
        """
        # Make sure that landmarkList is a list
        if not(type(landmarkList) is list): raise Exception("landmarkList must be of type list")
        
        if len(landmarkList) != 0:
            # Make sure that each landmark in landmarkList is a list
            for landmark in landmarkList:
                if not(type(landmark) is list): raise Exception("Each landmark in landmarkList must be a list.")
            
            # Make sure dimension of 1st landmark in landmarkList > 0
            dimension = len(landmarkList[0]) - 1
            if dimension < 1: raise Exception("Each landmark in landmarkList must have 1 or more dinemsions")

            for landmark in landmarkList:
                # Make sure each landmark in landmarkList has same dimension
                if len(landmark)-1 != dimension: raise Exception("Each landmark in landmarkList must have same dimension.")

                # Make sure each landmark in landmarkList has numeric cordinates
                for value in landmark[1:]:
                    try:
                        value + 0
                    except TypeError:
                        raise Exception("Coordinates of landmarks must be numeric.")

        self.landmarkList = landmarkList

    def GetLandmarks(self,labelList=[]):
        """
        Returns list of lanmarks in the format
            [[label1,x1,y1,...],...,[labelN,xN,yN,...]]
        """
        if labelList==[]:
            return self.landmarkList[:]
        else:
            outLandmarkList = []
            for label in labelList:
                for landmark in self.landmarkList:
                    if landmark[0] == label: outLandmarkList.append(landmark)
            return outLandmarkList

    def GetPoints(self, labelList=[]):
        """
        Returns list of lanmarks in the format
            [[x1,y1,...],...,[xN,yN,...]]
        """
        if labelList==[]:
            return self.landmarkList[:]
        else:
            outPointList = []
            for label in labelList:
                for landmark in self.landmarkList:
                    if landmark[0] == label: outPointList.append(landmark[1:])
            return outPointList

    
    def SetLabels(self,labelList):
        """
        Sets the labels of the landmarks where labelList must have format...
             [label1,...,labelN]
        It is necessary that len(labelList) == landmarks.GetNumberOfLandmarks()
        """
        if not(type(labelList) is list): raise Exception("labelList must be of type list")
        if len(labelList) != self.GetNumberOfLandmarks(): raise Exception("Label list must contain same number of labels as this landmarks object.")
        for i in range(0,self.GetNumberOfLandmarks()): self.landmarkList[i][0] = labelList[i]

    def GetLabels(self):
        """
        Returns list of landmark labels in the format...
            [label1,...,labelN]
        """
        labelList = []
        for landmark in self.landmarkList: labelList.append(landmark[0])
        return labelList

    
    def GetDimensionOfLandmarks(self):
        """
        Returns the dimension of landmarks.
        """
        return len(self.landmarkList[0]) - 1

    def GetNumberOfLandmarks(self):
        """
        Returns the number of landmarks.
        """
        return len(self.landmarkList)
    
    def GetSize(self):
        """
        Returns size of bounding box at origin which contains all landmarks
        """
        dimension = self.GetDimensionOfLandmarks() 
        size = [0]*dimension
        for lmk in self.landmarkList:
            for i in range(dimension):
                if lmk[i+1] > size[i]: size[i] = lmk[i+1]
        return(size)

    def GetMin(self):
        """
        Returns minimum index of bounding box containing all landmarks
        """
        dimension = self.GetDimensionOfLandmarks() 
        minValue = [float('inf')]*dimension
        for lmk in self.landmarkList:
            for i in range(dimension):
                if lmk[i+1] < minValue[i]: minValue[i] = lmk[i+1]
        return minValue

    def GetMax(self):
        """
        Returns maximum index of bounding box containing all landmarks
        """
        dimension = self.GetDimensionOfLandmarks() 
        maxValue = [float('-inf')]*dimension
        for lmk in self.landmarkList:
            for i in range(dimension):
                if lmk[i+1] > maxValue[i]: maxValue[i] = lmk[i+1]
        return maxValue


    def Read(self,path,spacing=[1.0,1.0,1.0]):
        """
        Reads landmarks from given path.
        It accepts landmarks files in the following formats
        1. CIS format which usually has extension .lmk
            Landmarks...
            N
            label1
            x1 y1 z1 1 1
            ...
            labelN
            xN yN zN 1 1
        2. DiffeoMap format which usually has extension .txt
            Landmarks...
            N
            label1 x1 y1 z1 ...
            ...
            labelN xN yN zN ...
        """
        self.spacing = spacing

        # Read lines from landmark file into list while triming whitespace and skiping blank line
        landmarkFile = open(path,"r")
        lineList = []
        for line in landmarkFile.readlines():
            if line.strip() != "": lineList.append(line.strip())
        landmarkFile.close()

        # Make sure this is a valid landmarks file by checking that the 1st line begins with "Landmarks"
        if lineList.pop(0)[0:9] != "Landmarks":
            raise Exception(path + " is not a valid landmarks file.")

        # Get number of landmarks from 2nd line of file
        numberOfLandmarks = int(lineList.pop(0))

        # Split all lines in line list into words
        #def StringSplit(string): return string.split()
        lineList = map(str.split,lineList)

        # Read landmarks from line list  
        landmarkList = []
        i = 0
        while i < len(lineList):
            if len(lineList[i]) == 1: # For CIS format landmarks
                label = lineList[i][0]
                index = map(float,lineList[i+1][0:3])
                i += 2
            else:                     # For DiffeoMap format landmarks
                label = lineList[i][0]
                index = map(float,lineList[i][1:4])
                i += 1

            point = [a*b for (a,b) in zip(index, spacing)]
            landmark = [label] + point
            landmarkList.append(landmark)
            if len(landmarkList) >= numberOfLandmarks: break

        self.SetLandmarks(landmarkList)

    def Write(self,path):
        """
        Writes landmarks to given path using CIS format.
        """
        # Write header
        landmarkFile = open(path,"w")
        print("Landmarks-1.0",file=landmarkFile)              # 1st line should always be "Landmarks-1.0"
        print(self.GetNumberOfLandmarks(), file=landmarkFile) # 2nd line is always the number of landmarks

        # Write landmarks
        for landmark in self.landmarkList:
            label = landmark[0]
            point = landmark[1:]
            index = [str(a/b) for (a,b) in zip(point, self.spacing)]
            print(label,file=landmarkFile)
            print(" ".join(index + ["1","1"]), file=landmarkFile)

        # Write tail
        # tail = '0\n0\n0\n0,1,0\n0\n"NeuroData"\n0.1,0.9\n"Voxel"\n0,0,0\n' + ",".join(map(str,self.spacing))
        # print(tail, file=landmarkFile)
        landmarkFile.close()

    def WriteCsv(self, path):
        csvFile = open(path, 'w')
        csvWriter = csv.writer(csvFile)
        header = ['label','x','y','z']
        csvWriter.writerows([header] + self.landmarkList)
        csvFile.close()

    def GetDistances(self,otherLandmarks,spacing=[1,1,1]):
        """
        Returns a list of distances between corresponding landmarks of this landmarks object and another landmarks objects.
        """
        if self.GetNumberOfLandmarks() != otherLandmarks.GetNumberOfLandmarks(): raise Exception("Other landmarks must have same number of landmarks as this landmarks object.")

        otherLandmarkList = otherLandmarks.GetLandmarks()
        distanceList = []
        for i in range(0,self.GetNumberOfLandmarks()):
            sumOfSquaredDifferences = 0
            for j in range(0,self.GetDimensionOfLandmarks()):
                sumOfSquaredDifferences += ((self.landmarkList[i][j+1] - otherLandmarkList[i][j+1])*spacing[j])**2

            distanceList.append(math.sqrt(sumOfSquaredDifferences))

        return distanceList

    def Affine(self, affine):
        if (not(type(affine)) is list) or (len(affine) != 12): raise Exception("affine must be a list of length 12.")
        lmkList = []
        labelList = self.GetLabels()
        for i in range(self.GetNumberOfLandmarks()):
            x0 = mat(self.landmarkList[i][1:]).reshape(3,1)
            A = mat(affine[:9]).reshape(3,3)
            b = mat(affine[9:]).reshape(3,1)
            #x1 = A.I*(x0 - b)
            x1 = A*x0 + b
            lmk = [labelList[i]] +  x1.flatten().tolist()[0]
            lmkList.append(lmk)
        return landmarks(lmkList, self.spacing)

    def Flip(self, size):
        """
        Returns flipped landmarks.
        Useful for converting BrainWorks Landmarks to MRIStudio Landmarks or vice versa.
        """
        if (not(type(size)) is list) or (len(size) != 3): raise Exception("size must be a list of length 3.")
        lmkList = []
        labelList = self.GetLabels()
        s = mat(size)
        for i in range(self.GetNumberOfLandmarks()):
            x0 = mat(self.landmarkList[i][1:])
            x1 = s - 1 - x0
            lmk  = [labelList[i]] + x1.tolist()[0]
            lmkList.append(lmk)
        return landmarks(lmkList, self.spacing)
    
    """
    def Resample(self, inSpacing, outSpacing):
        outLandmarkList = []
        for inLandmark in self.landmarkList:
            inLabel = inLandmark[0]
            inIndex = mat(inLandmark[1:])
            x = multiply(inIndex, inSpacing)
            outIndex = x / outSpacing
            outLandmark = [inLabel] + outIndex.tolist()[0]
            outLandmarkList.append(outLandmark)

        return landmarks(outLandmarkList)
    """

    def Crop(self, size):
        if (not(type(size)) is list) or (len(size) != 3): raise Exception("size must be a list of length 3.")
        outLandmarkList = []
        for inLandmark in self.landmarkList:
            inLabel = inLandmark[0]
            inCoordinate = mat(inLandmark[1:])
            outCoordinate = inCoordinate - mat(size)
            outLandmark = [inLabel] + outCoordinate.tolist()[0]
            outLandmarkList.append(outLandmark)

        return landmarks(outLandmarkList, self.spacing)

