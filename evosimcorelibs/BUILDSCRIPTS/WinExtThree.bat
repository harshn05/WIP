SET PATH=%PATH%;X:\python64\win3\
SET PATH=%PATH%;X:\python64\win3\Scripts
SET PATH=%PATH%;X:\python64\win3\Lib\site-packages\pywin32_system32
SET PATH=%PATH%;X:\python64\win3\share\mingwpy\bin 
del *~
del .fuse*
cd ..
python cysetup.py build_ext -i --compiler=mingw32
rmdir build /s /q
del *~
del .fuse*
cd ..
python EvoSim.py
