#!/bin/bash
export PATH="/media/Harsh/python64/lin3/bin:$PATH"
rm *~
rm .fuse*
cd ..
python cysetup.py build_ext -i
rm -rf build
rm *~
rm .fuse*
cd ..
cd uicomponents
rm resourceLIST_rc.py resourceLIST.py resourceLIST.pyc
pyrcc4 -py3 resourceLIST.qrc -o resourceLIST_rc.py 
cp resourceLIST_rc.py resourceLIST.py
cp resourceLIST_rc.py ../
cd ..
python EvoSim.py
