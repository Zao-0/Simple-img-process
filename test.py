# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 22:26:04 2020

@author: mika_
"""
import cv2

image = cv2.imread('miku.jpg')
cv2.imshow('picture', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
