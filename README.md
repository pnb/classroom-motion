# Classroom motion estimation
This software estimates camera pan, zoom, and instructor motion from videos of a classroom in uncontrolled conditions (i.e., a consumer-grade camera was placed at the back of an ordinary university lecture hall).

Methodological details will be described in a future publication.

## Requirements
* Python 3.6
* OpenCV 3 with Python bindings.
* Numpy
* SciPy
* Pandas

This software has only been tested on OSX, but should work fine on other operating systems. Lower versions of Python 3 (like 3.5) should work, but Python 2.7 has not been tested and is unlikely to work.

## Usage
Two Python scripts are included, one for estimating motion and one for estimating camera pan and zoom. Both scripts have some parameters at the top of the script that can be adjusted. `VISUALIZE_MOTION` and `VISUALIZE_POINTS` might both be best set to `False`, for example, to speed up processing if you are not interested in seeing the visualization.

##### Motion estimation script:
`python motion_detection.py input_video_filename output_csv_filename`

Acceptable input formats are determined by your particular OpenCV installation. Output will be in comma-separated values form.

##### Camera pan/zoom estimation script:
`python optical_flow_points.py input_video_filename output_csv_filename`

The arguments are the same as the motion estimation script.

Outputs of the two scripts for a single video should have the same number of lines, and can be aligned directly.
