import os 
os.system(f"git lfs install")
os.system(f"git clone https://gitlab.yephome.io/bigdata/dreamhome.git /home/demo/source/dreamhome")
os.chdir(f"/home/demo/source/dreamhome")
os.system(f"pip install -r requirements.txt")
os.system("python Demo.py")