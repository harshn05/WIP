SET PATH=%PATH%;X:\python64\win3\
SET PATH=%PATH%;X:\python64\win3\Scripts
SET PATH=%PATH%;X:\python64\win3\Lib\site-packages\pywin32_system32
SET PATH=%PATH%;X:\python64\win3\share\mingwpy\bin 
conda update anaconda
conda update conda
conda update --all
conda install mpmath pyqt=4.10.4 pyqtgraph
conda install -c https://conda.binstar.org/menpo opencv3
pip install -i https://pypi.binstar.org/carlkl/simple mingwpy