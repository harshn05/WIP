#!/bin/bash
export PATH="/media/Harsh/interpreters3/LinX64/bin:$PATH"
cd ..
python DeployScripts/pyinstaller3/pyinstaller.py -Fw EvoSim.py
