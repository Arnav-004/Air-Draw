import cv2
import mediapipe as mp
import numpy as np
import math
from Track_Hand import HandDetector

def main():
    vid = cv2.VideoCapture(0)
    detection = HandDetector(n_hands=1, min_detect_confidence=0.9, min_track_confidence=0.9)

    # canvas
    canvas = np.zeros((750, 1100, 3), dtype=np.uint8)

    prev = (-1,-1)
    curr = (-1,-1)
    
    while vid.isOpened:
        playing, frame = vid.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (1100, 750))

        if not playing:
            break

        # -----------------------------------------------------

        list = detection.findPos(frame, draw=False)

        if len(list) != 0 :
            draw = math.dist(list[4], list[12])
            erase = math.dist(list[4], list[8])
            if draw <= 50 :
                prev = curr
                curr = list[8]
                if prev != (-1, -1):
                    cv2.line(canvas, prev, curr, (255, 255, 255), 25, cv2.LINE_4) 
                    cv2.putText(frame, "DRAW", (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,0), 2, cv2.LINE_8)
            elif erase <= 50 :
                cv2.circle(canvas, list[17], 22, (0,0,0), -1, cv2.LINE_4)
                cv2.circle(canvas, list[18], 22, (0,0,0), -1, cv2.LINE_4)
                cv2.circle(canvas, list[19], 22, (0,0,0), -1, cv2.LINE_4)
                cv2.circle(canvas, list[20], 22, (0,0,0), -1, cv2.LINE_4)
                cv2.putText(frame, "ERASE", (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,0), 2, cv2.LINE_8)
            else:
                curr = (-1, -1)
        else:
            curr = (-1, -1)

        # -----------------------------------------------------

        res = cv2.addWeighted(frame, 1, canvas, 1, 5)

        # show the image
        # cv2.imshow('hand tracking', frame)
        # cv2.imshow('canvas', canvas)
        cv2.imshow('res', res)
        if cv2.waitKey(1) & 0xff == ord(' '):
            break

    vid.release()
    cv2.destroyAllWindows()
    

main()
