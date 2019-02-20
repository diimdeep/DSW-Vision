#!/usr/bin/env python

'''
example to show optical flow estimation using DISOpticalFlow

USAGE: dis_opt_flow.py [<video_source>]

Keys:
 1  - toggle HSV flow visualization
 2  - toggle glitch
 3  - toggle spatial propagation of flow vectors
 4  - toggle temporal propagation of flow vectors
ESC - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

from time import sleep

import numpy as np
import cv2 as cv
import video


bins = np.arange(256).reshape(256,1)

def hist_curve(im):
    h = np.zeros((300,256,3))
    if len(im.shape) == 2:
        color = [(255,255,255)]
    elif im.shape[2] == 3:
        color = [ (255,0,0),(0,255,0),(0,0,255) ]
    for ch, col in enumerate(color):
        hist_item = cv.calcHist([im],[ch],None,[256],[0,256])
        cv.normalize(hist_item,hist_item,0,255,cv.NORM_MINMAX)
        hist=np.int32(np.around(hist_item))
        pts = np.int32(np.column_stack((bins,hist)))
        cv.polylines(h,[pts],False,col)
    y=np.flipud(h)
    return y

def hist_lines(im):
    h = np.zeros((300,256,3))
    if len(im.shape)!=2:
        print("hist_lines applicable only for grayscale images")
        #print("so converting image to grayscale for representation"
        im = cv.cvtColor(im,cv.COLOR_BGR2GRAY)
    hist_item = cv.calcHist([im],[0],None,[256],[0,256])
    cv.normalize(hist_item,hist_item,0,255,cv.NORM_MINMAX)
    hist=np.int32(np.around(hist_item))
    for x,y in enumerate(hist):
        cv.line(h,(x,0),(x,y),(255,255,255))
    y = np.flipud(h)
    return y


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return bgr


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv.remap(img, flow, None, cv.INTER_LINEAR)
    return res


if __name__ == '__main__':
    import sys
    print(__doc__)
    try:
        fn = sys.argv[1]
    except IndexError:
        fn = 0

    cam = video.create_capture(fn)
    cam.set(cv.CAP_PROP_FRAME_WIDTH, 320)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
    sleep(1)
    ret, prev = cam.read()
    prevgray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
    show_hsv = False
    show_glitch = False
    use_spatial_propagation = False
    use_temporal_propagation = True
    cur_glitch = prev.copy()
    inst = cv.DISOpticalFlow.create(cv.DISOPTICAL_FLOW_PRESET_MEDIUM)
    inst.setUseSpatialPropagation(use_spatial_propagation)

    flow = None

    hsv_map = np.zeros((180, 256, 3), np.uint8)
    h, s = np.indices(hsv_map.shape[:2])
    hsv_map[:,:,0] = h
    hsv_map[:,:,1] = s
    hsv_map[:,:,2] = 255
    hsv_map = cv.cvtColor(hsv_map, cv.COLOR_HSV2BGR)
    cv.imshow('hsv_map', hsv_map)

    cv.namedWindow('hist', 0)
    hist_scale = 10
    def set_scale(val):
        global hist_scale
        hist_scale = val
    cv.createTrackbar('scale', 'hist', hist_scale, 32, set_scale)

    while True:
        ret, img = cam.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if flow is not None and use_temporal_propagation:
            #warp previous flow to get an initial approximation for the current flow:
            flow = inst.calc(prevgray, gray, warp_flow(flow,flow))
        else:
            flow = inst.calc(prevgray, gray, None)
        prevgray = gray

        cv.imshow('flow', draw_flow(gray, flow))
        im = draw_hsv(flow)
        if show_hsv:
            cv.imshow('flow HSV', im)
        if show_glitch:
            cur_glitch = warp_flow(cur_glitch, flow)
            cv.imshow('glitch', cur_glitch)

        ch = 0xFF & cv.waitKey(5)
        if ch == 27:
            break
        if ch == ord('1'):
            show_hsv = not show_hsv
            print('HSV flow visualization is', ['off', 'on'][show_hsv])
        if ch == ord('2'):
            show_glitch = not show_glitch
            if show_glitch:
                cur_glitch = img.copy()
            print('glitch is', ['off', 'on'][show_glitch])
        if ch == ord('3'):
            use_spatial_propagation = not use_spatial_propagation
            inst.setUseSpatialPropagation(use_spatial_propagation)
            print('spatial propagation is', ['off', 'on'][use_spatial_propagation])
        if ch == ord('4'):
            use_temporal_propagation = not use_temporal_propagation
            print('temporal propagation is', ['off', 'on'][use_temporal_propagation])
        if ch == ord('a'):
            curve = hist_curve(im)
            cv.imshow('histogram',curve)
            cv.imshow('image',im)
            print('a')
        elif ch == ord('b'):
            print('b')
            lines = hist_lines(im)
            cv.imshow('histogram',lines)
            cv.imshow('image',gray)
        elif ch == ord('c'):
            print('c')
            equ = cv.equalizeHist(gray)
            lines = hist_lines(equ)
            cv.imshow('histogram',lines)
            cv.imshow('image',equ)
        elif ch == ord('d'):
            print('d')
            curve = hist_curve(gray)
            cv.imshow('histogram',curve)
            cv.imshow('image',gray)
        elif ch == ord('e'):
            print('e')
            norm = cv.normalize(gray, gray, alpha = 0,beta = 255,norm_type = cv.NORM_MINMAX)
            lines = hist_lines(norm)
            cv.imshow('histogram',lines)
            cv.imshow('image',norm)
        elif ch == ord('f'):
            small = cv.pyrDown(im)

            hsv = cv.cvtColor(small, cv.COLOR_BGR2HSV)
            dark = hsv[...,2] < 32
            hsv[dark] = 0
            h = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

            h = np.clip(h*0.005*hist_scale, 0, 1)
            vis = hsv_map*h[:,:,np.newaxis] / 255.0
            cv.imshow('hist', vis)
    cv.destroyAllWindows()
