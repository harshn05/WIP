# -*- mode: python -*-

block_cipher = None


a = Analysis(['EvoSim.py'],
             pathex=['/media/Harsh/EvoSimPy/WIP'],
             hiddenimports=[],
             hookspath=None,
             runtime_hooks=None,
             excludes=None,
             cipher=block_cipher)
pyz = PYZ(a.pure,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='EvoSim',
          debug=False,
          strip=True,
          upx=True,
          console=False )
