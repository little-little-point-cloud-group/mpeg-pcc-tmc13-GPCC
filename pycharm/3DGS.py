from Player import GS_render
import os
MPEG_PCC_Render=r'PccAppRenderer.exe'
main = MPEG_PCC_Render
para1 = "-f  " + r"C:\little_boy\PCC_sequence\3DGS\m71763_breakfast_stable\track\frame000.ply"
para2 = "-g 1 "
para4 = "-y " + r"C:\little_boy\PCC_sequence\3DGS\m71763_breakfast_stable\cameras\frame000\cameras.txt"
para = "%s %s %s %s" % (main, para1, para2, para4)
para = para
os.system(para)