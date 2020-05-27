# Hand-Gesture Recognition

This project is for counting number of fingers shown to the camera. Though it can be developed using deep learning but I approached it using OpenCV and some basic mathematical operations. Few things which I learned implementing this project includes:
1. Background Subtration
2. Contour Detection
3. Thresholding, Erosion and Dilation

## Steps Involved

1. Background Subtraction
2. Motion Detection and Thresholding
3. Segment the hand using Contour Extraction
4. Finding convex hull of the hand
5. Comute center of palm
6. Using the palm’s center, construct a circle with the maximum Euclidean distance (between the palm’s center and the extreme points) as radius.
7. Perform bitwise AND operation between the thresholded hand image (frame) and the circular ROI (mask). This reveals the finger slices, which could further be used to calcualate the number of fingers shown.

## Requirements

1. [OpenCV](https://pypi.org/project/opencv-python/)
2. [Numpy](https://pypi.org/project/numpy/)
3. [Scikit-Learn](https://pypi.org/project/scikit-learn/)
4. [Python](https://www.python.org/downloads/)

## Example

See the below examples for demonstration.

<img src= '2.png'>
## Try it yourself

Type below instructions on command prompt:
```
git clone https://github.com/Aditya-Gupta1/hand-gesture-recognizer.git
cd hand-gesture-recognizer
python example.py
```
As the project is developed in object-oriented manner, we can test it in just 3 lines of code. Below is the code for `example.py` which you saw in above instructions.
```python
from handgesturerecognizer import HandGestureRecognizer
recognizer= HandGestureRecognizer()
recognizer.start(frame_rate= True, display_thresholded= True)
```
There are many customizable options in the start method. You can see the [source code](https://github.com/Aditya-Gupta1/hand-gesture-recognizer/blob/master/handgesturerecognizer.py) for more details.

## Conclusion

As this project involves no use of deep learning techniques, there is quite a large room for improvements. It is just a attempt by me to demonstrate my computer vision skills using nothing but opencv library.<br>
Thanks.