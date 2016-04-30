#!/bin/bash
SET PATH=%PATH%;X:\interpreters2\WinX64\share\mingwpy\bin
SET PATH=%PATH%;X:\interpreters2\WinX64\
SET PATH=%PATH%;X:\interpreters2\WinX64\Scripts
SET PATH=%PATH%;X:\interpreters2\WinX64\Lib\site-packages\pywin32_system32
cd ..
python DeployScripts/pyinstaller2/pyinstaller.py -Fw EvoSim.py
