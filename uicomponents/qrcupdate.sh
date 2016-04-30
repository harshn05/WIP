rm resourceLIST_rc.py resourceLIST.py resourceLIST.pyc
pyrcc4 -py3 resourceLIST.qrc -o resourceLIST_rc.py 
cp resourceLIST_rc.py resourceLIST.py
cp resourceLIST_rc.py ../
