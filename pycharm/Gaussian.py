
import os
import torch
from pathlib import Path
from tqdm import tqdm
from example.gs_quantize import quantize_3dg, dequantize_3dg
from example.gs_read_write import writePreprossConfig, readPreprossConfig, read3DG_ply, write3DG_ply
import matplotlib.pyplot as plt
import numpy as np



class GaussianModel:
    def __init__(self,path):
        file_raw = Path(path)  # input: the raw model frame in INRIA format

        # Modify these to the desired quantization parameters
        bits_pos = 18
        bits_sh = 12
        bits_opacity = 12
        bits_scale = 12
        bits_rot = 12

        limits_pos = [[0, 0, 0], 256]
        limits_sh = [-4, 4]
        limits_opacity = [-7, 18]
        limits_scale = [-26, 4]
        limits_rot = [-1, 1]

        bits = [bits_pos, bits_sh, bits_opacity, bits_scale, bits_rot]
        limits = [limits_pos, limits_sh, limits_opacity, limits_scale, limits_rot]

        # Quantization
        print("-( Read PLY )-------------------------")
        pos, sh, opacity, scale, rot = read3DG_ply(file_raw, tqdm)

        for k in range(3):
            limits[0][0][k] = pos[:, k].min()

        print("-( Quantize )-------------------------")
        self._xyz, self.sh, self._opacity, self._scaling, self._rotation = quantize_3dg(bits, limits, pos, sh, opacity, scale, rot, tqdm)

    def plot_all_heatmaps(self, bins=100, cmap='viridis', dpi=150):
        """生成所有属性组合的热力图"""
        # 定义属性字典（修正重复键）
        attributes = {
            "r": self.sh[:, 0, 0],
            "g": self.sh[:, 0, 1],
            "b": self.sh[:, 0, 2],
            "opacity": self._opacity,
            "scaling": self._scaling[:, 0],
            "rotation": self._rotation[:, 0]
        }
        # 创建输出目录
        os.makedirs("heatmaps", exist_ok=True)

        # 遍历字典中的每个数组
        for label, data in attributes.items():
            # 创建一个图和一个子图
            fig, ax = plt.subplots()
            ax.hist(data, bins=10, alpha=1, label=label, density=True)
            # 添加图例
            ax.legend()

            # 添加标题和坐标轴标签
            ax.set_title("统计折线图")
            ax.set_xlabel("索引")
            ax.set_ylabel("值")
            plt.xlim([0, 4000])
            plt.savefig("heatmaps" + "/" + label + ".png", bbox_inches='tight')
            plt.close()
            plt.clf()

        # 自动确定分箱数
        def auto_bins(data_length):
            return min(200, max(50, int(np.sqrt(data_length) / 2)))

        # 遍历所有属性组合
        keys = list(attributes.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                attr1 = keys[i]
                attr2 = keys[j]

                data1 = attributes[attr1]
                data2 = attributes[attr2]

                # 处理不同维度的数据
                cols1 = data1.shape[1] if data1.ndim > 1 else 1
                cols2 = data2.shape[1] if data2.ndim > 1 else 1

                for c1 in range(cols1):
                    for c2 in range(cols2):
                        # 提取数据列
                        x = data1[:, c1] if cols1 > 1 else data1.flatten()
                        y = data2[:, c2] if cols2 > 1 else data2.flatten()

                        # 数据采样（超过50万点时）
                        if len(x) > 5e5:
                            idx = np.random.choice(len(x), 500000, replace=False)
                            x, y = x[idx], y[idx]

                        # 动态调整分箱数
                        actual_bins = 5 * auto_bins(len(x))

                        # 创建热力图
                        plt.figure(figsize=(12, 8), dpi=dpi)

                        hh = plt.hist2d(x, y, bins=actual_bins, cmap=cmap, norm=LogNorm(vmin=1, vmax=None),
                                        density=True)

                        # 添加颜色条
                        cbar = plt.colorbar(hh[3])
                        cbar.set_label('Density (log scale)', rotation=270, labelpad=20)

                        # 设置标签和标题
                        plt.title(f"Heatmap: {attr1} vs {attr2}\n"
                                  f"Bins: {actual_bins}x{actual_bins}")
                        plt.xlabel(f"{attr1}")
                        plt.ylabel(f"{attr2}")

                        # 优化网格和样式
                        plt.grid(True, alpha=0.3, linestyle=':')

                        # 保存文件
                        filename = f"heatmaps/{attr1}__vs__{attr2}_heatmap.png"
                        plt.savefig(filename, bbox_inches='tight')
                        plt.close()
                        plt.clf()
                        print(f"已生成: {filename}")

    @property
    def get_scaling(self):
        return np.exp(self._scaling)


    @property
    def get_opacity(self):
        return 1/(1+np.exp(self._opacity))

    def my_fliter(self,radio=70):
        # 创建掩码（Mask）过滤小 scaling 值
        self.S = np.prod(self.get_scaling, axis=1)
        threshold = np.percentile(self.S, radio)  # 70%分位数
        mask=self.S<threshold
        # 应用掩码过滤所有属性
        self._xyz, self.sh, self._opacity, self._scaling, self._rotation=self._xyz[mask,:], self.sh[mask,:], self._opacity[mask,:], self._scaling[mask,:], self._rotation[mask,:]

    def write(self,path):
        os.makedirs("dequantized", exist_ok=True)
        print("-( Write dequantized PC )-------------")
        file_dequantized = Path(path)  # output: PLY file of the dequantized decoded frame

        write3DG_ply(self._xyz, self.sh, self._opacity, self._scaling, self._rotation, True,file_dequantized , tqdm)




