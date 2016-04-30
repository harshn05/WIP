import platform
import subprocess
import sys


if platform.system() == 'Linux':
	if sys.version_info.major == 2:
		subprocess.call(['./lin2.sh'])
	else:
		subprocess.call(['./lin3.sh'])
		
elif platform.system() == 'Windows':
	if sys.version_info.major == 2:
		try:
			subprocess.call(['win2.bat'])
		except:
			subprocess.call(['./win2.bat'])
	else:
		try:
			subprocess.call(['win3.bat'])
		except:
			subprocess.call(['./win3.bat'])
