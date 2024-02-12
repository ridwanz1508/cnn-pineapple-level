# CNN-pineapple-level
A Simple web desktop for detecting pineapple ripeness level with use CNN method. result data from detecting can be saved automatically in PC folder. data will be shows in percentage of pineapple ripeness level.

*Installation and Usage*
1. If you have previous/other manually installed (= not installed via pip) version of OpenCV installed (e.g. cv2 module in the root of Python's site-packages), remove it before installation to avoid conflicts.
2. Make sure that your pip version is up-to-date (19.3 is the minimum supported version): pip install --upgrade pip. Check version with pip -V. For example Linux distributions ship usually with very old pip versions which cause a lot of unexpected problems especially with the manylinux format.
3. Select the correct package for your environment:
Do not install multiple different packages in the same environment. There is no plugin architecture: all the packages use the same namespace (cv2). If you installed multiple different packages in the same environment, uninstall them all with pip uninstall and reinstall only one package.

Packages for standard desktop environments :
- pip install opencv-python
- pip install tensorflow
- pip install numpy
- pip install python-tk
