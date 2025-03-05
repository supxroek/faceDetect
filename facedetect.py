#!/usr/bin/env python

import numpy as np
import cv2 as cv
import requests
from video import create_capture
from common import clock, draw_str

URL_LINE = 'https://notify-api.line.me/api/notify'
LINE_ACCESS_TOKEN = 'yf34O2iOaX96TznK2CmCC0xGJdDn3rDfZray6sdWBUt'
LINE_HEADERS = {'Authorization': 'Bearer ' + LINE_ACCESS_TOKEN}

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags=cv.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects

def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)

def main():
    import sys, getopt

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)

    cascade_fn = args.get('--cascade', "haarcascades/haarcascade_frontalcatface.xml")
    nested_fn = args.get('--nested-cascade', "haarcascades/haarcascade_frontalcatface.xml")

    cascade = cv.CascadeClassifier(cv.samples.findFile(cascade_fn))
    nested = cv.CascadeClassifier(cv.samples.findFile(nested_fn))

    cam = create_capture(video_src, fallback='synth:bg={}:noise=0.05'.format(cv.samples.findFile('lena.jpg')))

    while True:
        _ret, img = cam.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)

        t = clock()
        rects = detect(gray, cascade)
        vis = img.copy()
        draw_rects(vis, rects, (0, 255, 0))
        if not nested.empty():
            for x1, y1, x2, y2 in rects:
                roi = gray[y1:y2, x1:x2]
                vis_roi = vis[y1:y2, x1:x2]
                subrects = detect(roi.copy(), nested)
                draw_rects(vis_roi, subrects, (255, 0, 0))

                # บันทึกและส่งภาพไปยัง Line Notify
                file_name = 'detected_cat.png'
                cv.imwrite(file_name, vis)
                with open(file_name, 'rb') as img_file:
                    msg = {'message': 'พบเจ้าเหมียว!'}
                    response = requests.post(URL_LINE, headers=LINE_HEADERS, files={'imageFile': img_file}, data=msg)
                    if response.status_code == 200:
                        print("แจ้งเตือนสำเร็จ!")
                    else:
                        print(f"การแจ้งเตือนล้มเหลว: {response.status_code}")

        dt = clock() - t
        draw_str(vis, (20, 20), 'time: %.1f ms' % (dt * 1000))
        cv.imshow('facedetect', vis)

        if cv.waitKey(5) == 27:
            break

    print('Done')

if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
