from pathlib import Path
import numpy as np
import matplotlib.pylab as plt
from tqdm import tqdm
from gs_quantize import quantize_3dg, dequantize_3dg
from gs_read_write import writePreprossConfig, readPreprossConfig, read3DG_ply, write3DG_ply

# Modify these locations to reflect your machine
file_raw = Path("Path/to/raw/model.ply")            # input: the raw model frame in INRIA format
file_config = Path("Path/to/the/config/file.json")  # output: json file containing the informarion necessary to inverse the quantization
file_quantized = Path("Path/to/quantized/file.ply") # output: PLY file with quantized 3DG attributes

# Modify these to the desired quantization parameters
bits_pos = 18
bits_sh = 12
bits_opacity = 12
bits_scale = 12
bits_rot = 12

limits_pos = [[0,0,0], 256]
limits_sh = [-4,4]
limits_opacity = [-7,18]
limits_scale = [-26,4]
limits_rot = [-1,1]

bits = [bits_pos, bits_sh, bits_opacity, bits_scale, bits_rot]
limits = [limits_pos, limits_sh, limits_opacity, limits_scale, limits_rot]

# Quantization
print("-( Read PLY )-------------------------")
pos, sh, opacity, scale, rot = read3DG_ply(file_raw, tqdm)

for k in range(3):
    limits[0][0][k] = pos[:,k].min()
writePreprossConfig(file_config, bits, limits)
    
print("-( Quantize )-------------------------")
q_pos, q_sh, q_opacity, q_scale, q_rot = quantize_3dg(bits, limits, pos, sh, opacity, scale, rot, tqdm)

print("-( Write quantized PC )---------------")
write3DG_ply(q_pos, q_sh, q_opacity, q_scale, q_rot, False, file_quantized, tqdm)
