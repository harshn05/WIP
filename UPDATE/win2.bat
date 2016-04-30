SET PATH=%PATH%;X:\python64\win2\
SET PATH=%PATH%;X:\python64\win2\Scripts
SET PATH=%PATH%;X:\python64\win2\Lib\site-packages\pywin32_system32
SET PATH=%PATH%;X:\python64\win2\share\mingwpy\bin
conda update anaconda
conda update conda
conda update --all
conda install mpmath pyqt=4.10.4 pyqtgraph sip
conda install -c https://conda.anaconda.org/menpo mayavi
conda install -c https://conda.binstar.org/menpo opencv3
pip install -i https://pypi.binstar.org/carlkl/simple mingwpy