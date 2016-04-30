SET PATH=%PATH%;X:\python64\win2\
SET PATH=%PATH%;X:\python64\win2\Scripts
SET PATH=%PATH%;X:\python64\win2\Lib\site-packages\pywin32_system32
SET PATH=%PATH%;X:\python64\win2\share\mingwpy\bin
SET PATH=%PATH%;X:\ProgramFiles\TiCTeX\MiKTeX\miktex\bin 
del *~
del .fuse*
cd ..
python cysetup.py build_ext -i --compiler=mingw32
rmdir build /s /q
del *~
del .fuse*
cd ..
python EvoSim.py