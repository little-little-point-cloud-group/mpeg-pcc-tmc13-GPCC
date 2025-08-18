import math
import subprocess
import shutil
import os
import re
import sys
import copy

class File:
    def copy_file(ori_path,copy_path):
        is_have_ori=os.path.exists(ori_path)
        
        if not is_have_ori:
            print("can not find the ori_file that will be copied")
        is_have_target=os.path.exists(copy_path)     
        if not is_have_target:
            shutil.copy(ori_path,copy_path)
    
    def generate_file(target_path=" ",ori_path=" "):
        def makedir(path):
            if os.path.exists(path):
                print(path,"has exist")
                return False
            s=path.split("\\")
            path1=s[0]
            for i in range(1,len(s)-1):
                path1=path1+"\\"+s[i]
            if not os.path.exists(path1):
                makedir(path1)
            os.makedirs(path)
            return True
        
        if ori_path==" ":
            makedir(target_path)
        else:
            File.copy_file(ori_path,target_path)                      #generate_file(base【文件名如.txt】,ori_path=r'D:\vs\attrs.yaml') or generate_file(base【可以使文件夹】)
        
    def get_all_file_from_baseCatalog(name,ori_path):
      def add_AllPathFromBase_into_list(name,ori_path,list_in):
        list_out=copy.deepcopy(list_in)
        list_path=os.listdir(ori_path)
        for file in list_path:
            if file.find("~&")==0:
                continue
            path=ori_path+"\\"+file
            if os.path.isdir(path):
                list_tem=add_AllPathFromBase_into_list(name,path,[])
                n=len(list_tem)
                for i in range(n):
                    list_out.append(list_tem[i])
            else:
                if file.find(name)>=0:
                    list_out.append(path)
        return list_out
    
      list_in=[]
      list_in=add_AllPathFromBase_into_list(name,ori_path,list_in)
      if len(list_in)==1:
           return list_in[0]
      elif len(list_in)==0:
           return None
    
      return   list_in                   #list=get_all_file_from_baseCatalog("encoder",base)
  
    def getFile_of_include_string_in_context(base,filename,string):
        all_encoder_path=File.get_all_file_from_baseCatalog(filename,base)
        
        def string1_Has_string2(string1,string2):
            a=True
            for str2 in string2:
                b=False
                for str1 in string1:
                    if str1.find(str2)>=0:
                        b=True
                if not b:
                    a=False
            return a
            
        def get_file_has_string(path,names):
            if isinstance(names,str):
                names=[names]

            #doc文件则进入
            def check_keyword_in_docx(file_path, keyword,lineHasSring):
                    num=0
                    doc = docx.Document(file_path)
                    for para in doc.paragraphs:
                        if keyword in para.text:
                            num= num+1
                            my_para.append(para.text)
                    if num==0:
                        return lineHasSring
                    lineHasSring["路径"]=file_path.split(base)[1]
                    lineHasSring["字符"].append(keyword)
                    lineHasSring["自然段"].append(my_para)
                    lineHasSring["次数"].append(num)
                    return lineHasSring
            if path.split(".")[-1]=="doc" or path.split(".")[-1]=="docx"  :
                keys=["路径","字符","自然段","次数"]
                
                my_para=[]
                values=[[],[],[],[]]
                lineHasSring=dict(zip(keys,values)) 
                for name in names:
                    lineHasSring=check_keyword_in_docx(path,names[0],lineHasSring)
                if len(lineHasSring["字符"])==len(names):
                   return lineHasSring
                else:
                    return []


            #非doc并且非pdf，则进入下面的判断
            with open(path,'r') as file:
                lines=file.readlines()
            numline=len(lines)
            keys=["path","string","numLine"]
            values=[path,[],[]]
            lineHasSring=dict(zip(keys,values))
            for i in range(numline):
                line=lines[i]
                for name in names:
                    if line.find(name)>=0:
                        lineHasSring["string"].append(line)
                        lineHasSring["numLine"].append(i+1)
            if string1_Has_string2(lineHasSring["string"],names):
                return lineHasSring
            else:
                return []
            
        list_out=[]
        if isinstance(all_encoder_path,str):
            all_encoder_path=[all_encoder_path]
        for path in all_encoder_path:
            list0=get_file_has_string(path,string)
            if len(list0)>0:
                list_out.append(list0)
        return list_out       #list=File.getFile_of_include_string_in_context(base,"encoder","inferredDirectCodingMode: 2")
    
    def getUniqueContent_from_files_in_base(file,base):              
        files=File.get_all_file_from_baseCatalog(file,base)
        if len(files)==0:
            print("     error:can not find "+file)
            return False
        if len(files)==1:
            print("     error:it only includes a path called as "+files[0])
            return False
        def commonString(list1,list2):
            common=[]
            for ele1 in list1:
                for ele2 in list2:
                    if ele1==ele2 and (not ele1=="\n"):
                        common.append(ele1)
                        list2.remove(ele1)
                        break
            return common
        
        with open(files[0],'r') as path:
                    common=path.readlines()
        for i in range(1,len(files)):
            with open(files[i],'r') as path:
                    lines1=path.readlines() 
            common=commonString(common,lines1)   
            
        def get_unique(common,lines):
            result=dict()
            for ele2 in range(len(lines)):
                com=True
                for ele1 in common:
                    if ele1==lines[ele2] or (lines[ele2]=="\n") :
                        com=False
                if com:
                    result[ele2+1]=lines[ele2]
            return result
        
        result=dict()
        for i in range(len(files)):
            with open(files[i],'r') as path:
                    lines1=path.readlines()
            unique=get_unique(common,lines1)
            p=files[i].split(base)[1]
            result[p]=unique
        return result                   #文件获取、粘贴、复制
    
class YUVPlayer:

    class VideoProcessor:
            def __init__(self, filename):
                self.filename = filename
                self.pix_fmt = None
                self._detect_resolution()
                self._detect_pix_fmt()
                
            def _detect_resolution(self):
      
                # 匹配 数字x数字 格式（如 3840x2160）
                match = re.search(r'(\d+)x(\d+)', self.filename)
                if match:
                    self.width = int(match.group(1))
                    self.height = int(match.group(2))
                else:
                    # 若未找到则尝试默认值或报错
                    self.width = 1920
                    self.height = 1080
                    print(f"警告：文件名 {self.filename} 中未检测到分辨率，使用默认 1920x1080")

            def get_pix_fmts(self):
                try:
                    cmd = [r'ffmpeg\ffplay.exe', '-hide_banner', '-pix_fmts']
                    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    output = proc.stdout
                    print(output)
                except (subprocess.CalledProcessError, FileNotFoundError) as e:
                    print(f"Error获取像素格式列表: {e}")
                    return []

                pix_fmts = []
                start_parsing = False

                for line in output.split('\n'):
                    line = line.strip()
                    if line.startswith('Pixel formats:'):
                        start_parsing = True
                        continue
                    if not start_parsing or not line:
                        continue
        
                    parts = re.split(r'\s+', line)
                    if len(parts) >= 2:
                        fmt_name = parts[1]
                        pix_fmts.append(fmt_name)

                # 去重并按名称长度降序排列（优先匹配更长名称）
                pix_fmts = list({fmt: None for fmt in pix_fmts}.keys())  # 去重
                pix_fmts.sort(key=lambda x: -len(x))
                return pix_fmts

            def _detect_pix_fmt(self):
                
                pix_fmts = self.get_pix_fmts()
                if not pix_fmts:
                    print("警告：无法获取像素格式列表，使用默认值yuv420p")
                    self.pix_fmt = 'rgb48le'
                    return

                # 优先检查完整单词匹配（使用正则表达式）
                filename_lower = self.filename.lower()
                for fmt in pix_fmts:
                    pattern = r'\b' + re.escape(fmt.lower()) + r'\b'
                    if re.search(pattern, filename_lower):
                        self.pix_fmt = fmt
                        return

                # 如果未找到单词匹配，尝试子字符串匹配
                for fmt in pix_fmts:
                    if fmt.lower() in filename_lower:
                        self.pix_fmt = fmt
                        return

                # 最终未找到则使用默认值
                self.pix_fmt = 'rgb48le'
                print(f"警告：未在文件名 {self.filename} 中发现像素格式，已设为默认rgb24")

    def __init__(self,path):
        self.rawvideo=path
        self.ffmpeg_play = r'ffmpeg\ffplay.exe'       # FFmpeg可执行路径
        self.ffmpeg_path=r'ffmpeg\ffmpeg.exe'
        self.yuv_file =  path       # YUV文件路径
        h_video=self.VideoProcessor(path)
        self.width = h_video.width                 # 视频宽度
        self.height = h_video.height                   # 视频高度
        self.pix_fmt = h_video.pix_fmt         # YUV像素格式
        self.framerate = 10                # 帧率

    def play(self):
        # 组装FFmpeg命令参数
        main = self.ffmpeg_play
        input_params = f" -video_size {self.width}x{self.height} -pixel_format {self.pix_fmt} -framerate {self.framerate}"
        input_file = f"-i \"{self.yuv_file}\""

        # 拼接完整命令
        command = f"{main} {input_params} {input_file}"
        
        # 执行命令
        os.system(command)

    def save_frames(self, start_frame: int, num_frames: int, output_dir: str = ""):
        """
        使用 FFmpeg 截取视频中的图片帧。
        
        :param start_frame: 截取开始的帧数
        :param num_frames: 截取的帧数
        :param output_dir: 图片输出目录
        """
        # 如果没有指定输出目录，则使用当前目录
        if not output_dir:
            output_dir = os.getcwd()
        
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        v = self.rawvideo.split('\\')[-1].split('_')[0]
        # 设置输出图像文件的路径（例如 frame_0001.png）
        output_path = os.path.join(output_dir, v+"_frame_%04d.jpg")
        
        # 构建 FFmpeg 命令
        command = [
            self.ffmpeg_path,             # ffmpeg 的路径
            '-video_size', f'{self.width}x{self.height}',  # 视频分辨率
            '-pixel_format', self.pix_fmt,  # 像素格式
            '-framerate', str(self.framerate),  # 帧率
            '-i', self.yuv_file,         # 输入 YUV 文件
            '-vf', f'select=between(n\,{start_frame}\,{start_frame + num_frames - 1}),setpts=N/FRAME_RATE/TB',  # 选择要提取的帧
            '-f', 'image2',              # 输出格式为图像序列
            '-vframes', str(num_frames),  # 提取的帧数
            output_path                  # 输出文件路径
        ]
        
        try:
            # 执行 FFmpeg 命令
            subprocess.run(command, check=True)
            print(f"Successfully saved {num_frames} frames to {output_dir}")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while extracting frames: {e}")

class GS_render:
    def __init__(self,path):
        base=os.getcwd()
       
        viewer_path=r'viewers\bin\SIBR_gaussianViewer_app.exe'
        MPEG_PCC_Render=r'PccAppRenderer.exe'
        self.viewer_path=viewer_path
        self.MPEG_PCC_Render=MPEG_PCC_Render
        self.path=path

    def save_image(self, yuv_paths="", start_frame=0, num_frames=1):      #yuv_paths放yuv视频文件夹
        '''
        ....\\yuv_paths
            ||--XXX.yuv
            ||--XXX.yuv
            其中XXX中需要包含视频的长宽高，格式，目前只支持v11_texture_1920x1080_yuv420p10le.yuv格式，v+数字_XXX_height x width_视频格式.yuv(最后必须是.yuv)
        '''

        files=os.listdir(yuv_paths)
        for f in files:
            h=YUVPlayer(yuv_paths+"\\"+f)
            h.save_frames(start_frame, num_frames, output_dir=self.path+"\\input")

    def _3DGS_show(self,paths=False):
         main=self.viewer_path
      
         if paths==False:
             para1="-m  "+self.path
             para="%s %s"%(main,para1)
             os.system(para)
         else:
             files=os.listdir(paths)
             for file in files:
                 path=paths+"\\"+file
                 para1="-m "+path
                 para="%s %s"%(main,para1)
                 os.system(para)

    def _help(self):
         main=self.viewer_path
         para1="-help"
         para="%s %s"%(main,para1)
         os.system(para)              #3DGS的显示工具

    def _MPEG_PCC_Render(self,path=False):
            
             main=self.MPEG_PCC_Render
             files=File.get_all_file_from_baseCatalog("point_cloud.ply",self.path)
             print("点云："+files[1])
             para1="-f  "+files[1]
             if path:
                para1="-f  "+path
             para2="-g 1 "
             para3="-o "+self.path+"/h"
             para4="-x "+r'C:\Users\31046\Desktop\3DGS\mirror\output\cameras.json'
             para="%s %s %s %s %s"%(main,para1,para2,para3,para4)
             para=para
             os.system(para)

    def training(self,sh_degree=3,convert=True):
        # 切换到目标目录
        os.chdir(r'C:\little_boy\vs_workspace\3dgs\gaussian-splatting')
        print(f"切换到目录: {os.getcwd()}")
        
        # 执行命令：python convert.py -s data
        cmd = [sys.executable, 'convert.py', '-s', self.path]
        if convert:
            print("执行命令:", ' '.join(cmd))
            subprocess.run(cmd, check=True)

        cmd = [sys.executable, 'train.py', '-s', self.path,"-m",self.path+"/output","--sh_degree",str(sh_degree)]
        print("执行命令:", ' '.join(cmd))
        subprocess.run(cmd, check=True)

        self.metrics()

    def metrics(self):
        # 切换到目标目录
        os.chdir(r'C:\little_boy\vs_workspace\3dgs\gaussian-splatting')
        print(f"切换到目录: {os.getcwd()}")
        
        # 执行命令：python convert.py -s data
        cmd = [sys.executable, 'render.py', '-m', self.path+"/output"]
        print("执行命令:", ' '.join(cmd))
        subprocess.run(cmd, check=True)

        cmd = [sys.executable, 'metrics.py', '-m', self.path+"/output"]
        print("执行命令:", ' '.join(cmd))
        subprocess.run(cmd, check=True)

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    # 1. 定义熵函数 H(P)
    def entropy(p):
        """计算二项分布的熵 H(P) = -P*log2(P) - (1-P)*log2(1-P)"""
        # 处理边界：P=0或1时，熵为0
        p = np.clip(p, 1e-12, 1 - 1e-12)  # 避免log(0)的情况
        p1=2.5*p/3.5
        p2=p/3.5
        p=1-p
        H=-p1 * np.log2(p1)-p2 * np.log2(p2)-p * np.log2(p)
        b1=p1*3
        b2=p2*5
        b3=p*8

        return H+b1+b2+b3


    # 1. 定义熵函数 H(P)
    def entropy1(p):
        """计算二项分布的熵 H(P) = -P*log2(P) - (1-P)*log2(1-P)"""
        # 处理边界：P=0或1时，熵为0
        p = np.clip(p, 1e-12, 1 - 1e-12)  # 避免log(0)的情况
        p1 = 2.5 * p / 3.5
        p2 = 0.2
        p = 1 - p
        H = -p1 * np.log2(p1) - p2 * np.log2(p2) - p * np.log2(p)
        b1 = p1 * 3
        b2 = p2 * 5
        b3 = p * 8

        return H + b1 + b2 + b3


    # 2. 生成P的取值范围（0到1之间，取1000个点）
    P = np.linspace(0, 1, 1000)

    # 3. 计算目标函数 H(P) - 3P
    y = entropy1(P) - 8

    # 4. 绘制曲线
    plt.figure(figsize=(8, 5))
    plt.plot(P, y, color='blue', linewidth=2,label="1")

    # 添加标题和坐标轴标签
    plt.title(r'H(P) - 8 的曲线图', fontsize=14)
    plt.xlabel(r'P', fontsize=12)
    plt.ylabel(r'bit', fontsize=12)

    # 3. 计算目标函数 H(P) - 3P
    y = entropy(P) - 8


    plt.plot(P, y, color='red', linewidth=2,label="2")

    # 添加标题和坐标轴标签
    plt.title(r'H(P) - 8 的曲线图', fontsize=14)
    plt.xlabel(r'P', fontsize=12)
    plt.ylabel(r'bit', fontsize=12)
    # 添加网格线
    plt.grid(alpha=0.3)

    # 显示图形
    plt.show()
    plt.legend(loc='upper right')

    plt.savefig( "比特.png", bbox_inches="tight", dpi=1200)