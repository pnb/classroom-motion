import sys
from collections import deque, OrderedDict
import cv2
import pandas as pd
from scipy import signal


VISUALIZE_MOTION = True


if len(sys.argv) != 3:
    raise AssertionError('Incorrect arguments: requires input_filename and output_filename')
infile = sys.argv[1]
outfile = sys.argv[2]
print('Initializing')


class MotionEstimator:
    def __init__(self, frame_rate, background_history_ms=3000, mask_time_ms=200, mask_size=10):
        self.frame_rate = frame_rate
        self.background_history_ms = background_history_ms
        self.mask_time_ms = mask_time_ms
        self.mask_size = mask_size
        self._mask_hist = deque(maxlen=int(mask_time_ms * frame_rate / 1000))
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=int(frame_rate * background_history_ms / 1000), detectShadows=False)
        self._dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, ksize=(mask_size, mask_size))
        self._first_frame = True

    def update(self, new_frame):
        # Subtract the background from the frame to get newly changed pixels.
        mask = self._bg_subtractor.apply(new_frame)
        raw_motion = cv2.countNonZero(mask) / (new_frame.shape[0] * new_frame.shape[0])

        # Keep only motion pixels that fall in the area around other recent motion pixels.
        smoothed = mask
        for m in self._mask_hist:
            smoothed = cv2.bitwise_and(smoothed, m)
        smooth_motion = cv2.countNonZero(smoothed) / (new_frame.shape[0] * new_frame.shape[0])

        # Dilate motion pixels to provide a region of possible motion for future frames.
        # A type of morphological filtering.
        dilated = cv2.dilate(mask, self._dilate_kernel, iterations=1)
        self._mask_hist.append(dilated)

        if self._first_frame:
            self._first_frame = False
            return 0.0, 0.0, smoothed  # Motion makes no sense in the first frame and is always 1.
        return raw_motion, smooth_motion, smoothed


cap = cv2.VideoCapture(infile)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) + 1  # A bit buggy, but better than nothing.
fps = cap.get(cv2.CAP_PROP_FPS)
ret, frame = cap.read()
if not ret:
    raise IOError('Could not open video file!')
frame_num = 1

estimators = []
for hist_ms in [500, 1000, 2000, 4000]:
    for smooth_ms in [100, 200]:
        estimators.append(MotionEstimator(fps, hist_ms, smooth_ms))

results = []
while ret:
    if frame_num % 100 == 0:
        print('Processing frame ' + str(frame_num) + ' of about ' + str(frame_count), end='\r')
    results.append(OrderedDict())
    results[-1]['video_file'] = infile
    results[-1]['fps'] = fps
    results[-1]['frame_num'] = frame_num
    results[-1]['sec_into_video'] = frame_num / fps

    for est in estimators:
        raw_motion, smooth_motion, mask = est.update(frame)
        results[-1]['raw_motion_' + str(est.background_history_ms)] = raw_motion
        results[-1]['smooth_motion_' + str(est.background_history_ms) + '_' +
                    str(est.mask_time_ms)] = smooth_motion

    if VISUALIZE_MOTION:
        cv2.imshow('frame', mask)
        cv2.waitKey(1)

    ret, frame = cap.read()
    frame_num += 1

print('\nSaving output')
results = pd.DataFrame.from_records(results)
results.to_csv(outfile, index=False)

cap.release()
cv2.destroyAllWindows()
