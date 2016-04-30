import platform
import subprocess
import sys


if platform.system() == 'Linux':
	if sys.version_info.major == 2:
		subprocess.call(['./LinuxExtTwo.sh'])
	else:
		subprocess.call(['./LinuxExtThree.sh'])
		
elif platform.system() == 'Windows':
	if sys.version_info.major == 2:
		try:
			subprocess.call(['WinExtTwo.bat'])
		except:
			subprocess.call(['./WinExtTwo.bat'])
	else:
		try:
			subprocess.call(['WinExtThree.bat'])
		except:
			subprocess.call(['./WinExtThree.bat'])
