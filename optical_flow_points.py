import itertools
import sys
from collections import deque, OrderedDict
import cv2
import numpy as np
import pandas as pd


FLOW_POINTS = 50
MIN_DISTANCE = 10  # Minimum distance required between tracking points.
VISUALIZE_POINTS = True


def get_moving_points(point_history):
    '''Find middle 50% of moving points, to exclude stuck points or large jumps. Ignores X vs Y.

    :param point_history: 2D Numpy array where each row is all tracked points at one timestep
    :returns: tuple with list of indices of the moving points, mean std. dev. of these points
    '''
    assert(np.shape(point_history)[0] < 1000)  # Make sure an entire history isn't passed in.
    stddevs = np.std(point_history, axis=0)
    devmap = {s: i for i, s in enumerate(stddevs)}
    key_points = [devmap[s] for s in sorted(devmap)][FLOW_POINTS // 4:FLOW_POINTS * 3 // 4]
    return key_points, np.mean(stddevs[key_points])


def detect_pan(point_history):
    '''Detect camera pans from a brief history of recent tracked optical flow points.

    :param point_history: 2D Numpy array where each row is all tracked points at one timestep
    :returns: tuple of (True/False for pan, stddev of moving points, correlation of moving points)
    '''
    point_i, stddev = get_moving_points(point_history)
    corr = np.corrcoef(point_history.T[point_i])
    mean_corr = np.mean(corr[np.triu_indices_from(corr, 1)])
    return stddev > .5 and mean_corr > .5, stddev, mean_corr


def detect_zoom(point_history, frame_shape):
    '''Detect camera zoom events from a brief history of recent tracked optical flow points.

    :param point_history: 2D Numpy array where each row is all tracked points at one timestep
    :param frame_shape: Tuple of the video frame shape
    :returns: tuple of (True/False for zoom, stddev of moving points, correlation of moving points)
    '''
    # Calculate distance of each point (combining X and Y) relative to the center of the frame.
    center = np.array(frame_shape) / 2
    new_hist = []
    for row in point_history:
        new_hist.append([])
        for xi in range(0, len(row), 2):
            new_hist[-1].append(np.linalg.norm(row[xi:xi + 2] - center))
    new_hist = np.array(new_hist)
    # Find points moving relative to the center, and determine if they are moving simultaneously
    # closer or further away from the center, which would indicate a zoom event.
    point_i, stddev = get_moving_points(new_hist)
    corr = np.corrcoef(new_hist.T[point_i])
    mean_corr = np.mean(corr[np.triu_indices_from(corr, 1)])
    return stddev > 1 and mean_corr > .9, stddev, mean_corr


if len(sys.argv) != 3:
    raise ValueError('Requires video_file_name and output_file_name arguments')
input_file = sys.argv[1]
output_file = sys.argv[2]

print('Initializing')
cap = cv2.VideoCapture(input_file)
ret, frame1 = cap.read()
if not ret:
    raise IOError('Could not open video file!')
fps = cap.get(cv2.CAP_PROP_FPS)
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

color = np.random.randint(0, 255, (FLOW_POINTS, 3))
mask = np.zeros_like(frame1)
old_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
old_points = cv2.goodFeaturesToTrack(old_gray, mask=None,
                                     maxCorners=FLOW_POINTS,
                                     qualityLevel=.01,
                                     minDistance=MIN_DISTANCE,
                                     blockSize=7)
results = [OrderedDict({'frame_num': 1})]  # First frame is not really processed.
point_history = deque(maxlen=int(fps))

for frame_num in itertools.count(2):
    ret, frame2 = cap.read()
    if not ret:
        break
    elif frame_num % 100 == 0:
        print('Processing frame ' + str(frame_num) + ' of about ' + str(num_frames), end='\r')

    new_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    new_points, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, old_points,
                                                   nextPts=None,
                                                   winSize=(15, 15),
                                                   maxLevel=3,
                                                   criteria=(cv2.TERM_CRITERIA_EPS |
                                                             cv2.TERM_CRITERIA_COUNT, 20, 0.03))
    need_replacing = FLOW_POINTS - sum(st.reshape(-1))
    if need_replacing > 0:  # Not all tracking points found, get some new ones.
        replacements = cv2.goodFeaturesToTrack(old_gray, mask=None,
                                               maxCorners=2 * FLOW_POINTS,  # Large pool.
                                               qualityLevel=.01,
                                               minDistance=MIN_DISTANCE,
                                               blockSize=7)
        for i in range(FLOW_POINTS):
            if not st[i]:
                for r in replacements:
                    for p in new_points:
                        if np.linalg.norm(r - p) < 10:  # Don't use if it is too close to another.
                            break
                    else:
                        new_points[i] = r

    if VISUALIZE_POINTS:
        for i, (new, old) in enumerate(zip(new_points, old_points)):
            a, b = new.ravel()
            c, d = old.ravel()
            if abs(a - c) > 50 or abs(b - d) > 50:
                continue
            mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            frame2 = cv2.circle(frame2, (a, b), 5, color[i].tolist(), -1)
        img = cv2.add(frame2, mask)
        cv2.putText(img, str(frame_num), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        cv2.imshow('frame', img)
        cv2.waitKey(1)

    old_gray = new_gray.copy()
    old_points = new_points.reshape(-1, 1, 2)
    point_history.append(old_points.reshape(-1))
    results.append(OrderedDict({'frame_num': frame_num}))
    results[-1]['tracking_points_replaced'] = need_replacing
    if len(results) >= fps:
        # Pan/zoom estimation lags behind, so insert results into the past.
        ndhist = np.array(point_history)
        pan, stddev, corr = detect_pan(np.array(ndhist))
        results[-int(fps / 2)]['is_pan'] = int(pan)
        results[-int(fps / 2)]['pan_point_stddev'] = stddev
        results[-int(fps / 2)]['pan_point_correlation'] = corr
        zoom, stddev, corr = detect_zoom(np.array(ndhist), new_gray.shape)
        results[-int(fps / 2)]['is_zoom'] = int(zoom and not pan)
        results[-int(fps / 2)]['zoom_point_stddev'] = stddev
        results[-int(fps / 2)]['zoom_point_correlation'] = corr

print('\nSaving results')
results = pd.DataFrame.from_records(results)
results.insert(0, 'video_file', input_file)
results.insert(1, 'fps', fps)
results.insert(3, 'sec_into_video', results.frame_num / fps)
results.to_csv(output_file, index=False)
cap.release()
cv2.destroyAllWindows()
