import argparse
import imutils
import cv2
import os


class ShotDetector:
    def __init__(self, video_path):
        self.video_path = video_path

    def detect_shot_boundary(self):
        fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()

        captured, total, frames = False, 0, 0
        count = 0

        vs = cv2.VideoCapture(self.video_path)
        (W, H) = (None, None)

        while True:
            (grabbed, frame) = vs.read()

            if frame is None:
                break

            orig = frame.copy()
            frame = imutils.resize(frame, width=600)
            mask = fgbg.apply(frame)

            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            if W is None or H is None:
                (H, W) = mask.shape[:2]

            p = (cv2.countNonZero(mask) / float(W * H)) * 100

            if p < args["min_percent"] and not captured and frames > args["warmup"]:

                cv2.imwrite("Captured" + str(total), frame)
                captured = True

                filename = "{}.png".format(total)
                path = os.path.sep.join(["shot_detection_results", filename])
                total += 1
                # save the  *original, high resolution* frame to disk
                print("[INFO] saving {}".format(path))
                cv2.imwrite(path, orig)

            elif captured and p >= args["max_percent"]:
                captured = False
            frames += 1


def main():
    ShotDetector(video_path="videos/vid1.mp4").detect_shot_boundary()
