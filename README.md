# Camera-Caliberaion
Caliberation.py does single camera caliberation.

StereoCaliberation.py does stereo caliberation, stereo rectify, remapping. The 13th pair is faulty, so I discard it when doing stereo caliberation.

Disparity.py does stereo matching by SGBM and computes the disparity map.
Parameters are copied from 
https://github.com/AlekMabry/StereoExamples/blob/c41ea3e7c9744d197957f9a550c59370c8115c6e/stereo_match.py

Numerical results are contained in Stereo Caliberate Results.
