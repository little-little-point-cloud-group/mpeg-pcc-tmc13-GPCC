from my_tools import File,Help_Statistics
import subprocess,os
import shutil

def run(name):
    main=r"C:\Users\31046\Desktop\IDCM\idcm\build\tmc3\Release\tmc3.exe"
    b=r'C:\Users\31046\Desktop\IDCM\idcm\build\test'
    para_encfg="-c " + b+"\\data\\"+name+"\\encoder.cfg"
    para2 = "--uncompressedDataPath=" + b+"\\data\\"+name+"\\quantized.ply"
    para3 = "--compressedStreamPath=" + b+"\\compress.bin"
    para4 = "--reconstructedDataPath=" + b+"\\"+name+".ply"
    para_e = "%s %s %s %s %s"%(main,para_encfg,para2,para3,para4)

    r=subprocess.run(para_e,capture_output=True,text=True)
    #print(r.stdout)


path=r'C:\Users\31046\Desktop\IDCM\idcm\build\test\test.txt'
base=r'C:\Users\31046\Desktop\IDCM\data_6近邻个数'
data=r"C:\Users\31046\Desktop\IDCM\idcm\build\test\data"
names=[#"basketball",
        "fruit",
       "breakfast",
       "cinema",
       "Palazzo",
       #"Staue",
       #"House",
       #"bicycle",
]
for name in names:
    #run(name)
    os.makedirs(base + "/" + name, exist_ok=True)
    #shutil.move(path,base+"/"+name+"/"+name+".txt")

    h=Help_Statistics(base+"/"+name,base+"/"+name+"/"+name+".txt",auto="上下文")
    #h.generate_PointCloud(outname="idcm",YUV=False,Class="C&level")
