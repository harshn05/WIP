import platform
import subprocess
import sys


if platform.system() == 'Linux':
	if sys.version_info.major == 2:
		subprocess.call(['./PACKTwo.sh'])
	else:
		subprocess.call(['./PACKThree.sh'])
		
elif platform.system() == 'Windows':
	if sys.version_info.major == 2:
		try:
			subprocess.call(['PACKTwo.bat'])
		except:
			subprocess.call(['./PACKTwo.bat'])
	else:
		try:
			subprocess.call(['PACKThree.bat'])
		except:
			subprocess.call(['./PACKThree.bat'])
