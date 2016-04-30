#!/bin/bash
export PATH="/media/Harsh/interpreters2/LinX64/bin:$PATH"
cd ..
python DeployScripts/pyinstaller2/pyinstaller.py -Fw EvoSim.py
