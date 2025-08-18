
import subprocess
import shutil
import os
import copy
import deepdiff
import openpyxl
import open3d
import numpy as np
import re
import matplotlib.pyplot as plt
import docx
import math
import statistics
import pandas as pd
import xlsxwriter
import random
import plotly.graph_objects as go
import filecmp
import tqdm

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


class Help_PointCloud:  # 观察点云的类，相对比较完善
    def __init__(self, path, path2=None):
        file_type = os.path.basename(path).split(".")[-1]
        if file_type == "ply":
            self.pointcloud = self._3Dread(path)
        else:
            print(path + "is not ply")
        self.pointcloud2 = False

        if not path2 == None:
            self.pointcloud2 = self._3Dread(path2)
        self.path = path
        self.path2 = path2

    def _3Dread(self, path):
        return open3d.io.read_point_cloud(path)

    def _write(pointcloud, path):
        open3d.io.write_point_cloud(path, pointcloud, write_ascii=True)

    def _setColor(self, color=None, color2=None):
        if not color == None:
            point_color = np.asarray(self.pointcloud.colors)
            for i in range(3):
                point_color[:, i] = color[i]
        if not color2 == None:
            if self.pointcloud2:
                point_color2 = np.asarray(self.pointcloud2.colors)
                for i in range(3):
                    point_color2[:, i] = color2[i]

    def _3Dshow(self, part_pointcloud=False, part_pointcloud2=False, name=None, background_color=[0, 0, 0],
                point_size=10):
        vis = open3d.visualization.Visualizer()
        if name == None:
            name = "pointcloud"
        vis.create_window(name, 1840, 1080)

        opt = vis.get_render_option()
        opt.point_size = point_size
        opt.background_color = background_color
        if part_pointcloud:
            aabb = part_pointcloud.get_axis_aligned_bounding_box()
            vis.add_geometry(part_pointcloud)
            vis.add_geometry(aabb)
            if part_pointcloud2:
                vis.add_geometry(part_pointcloud2)

            vis.run()
            return 0

        vis.add_geometry(self.pointcloud)
        if self.pointcloud2:
            vis.add_geometry(self.pointcloud2)

        vis.run()

    def show_3dBox(self, boxmin, boxmax):
        def fun(pcd):
            part_of_pointcloud = copy.deepcopy(pcd)
            points = np.asarray(part_of_pointcloud.points)
            x_filt = np.logical_and(
                (points[:, 0] > boxmin[0]), (points[:, 0] < boxmax[0]))
            y_filt = np.logical_and(
                (points[:, 1] > boxmin[1]), (points[:, 1] < boxmax[1]))
            z_filt = np.logical_and(
                (points[:, 2] > boxmin[2]), (points[:, 2] < boxmax[2]))
            filt = np.logical_and(x_filt, y_filt)  # 必须同时成立
            filt = np.logical_and(filt, z_filt)  # 必须同时成立
            points = points[filt, :]  # 过滤
            color = np.asarray(part_of_pointcloud.colors)
            color = color[filt, :]
            #
            part_of_pointcloud.points = open3d.utility.Vector3dVector(points)
            part_of_pointcloud.colors = open3d.utility.Vector3dVector(color)
            return part_of_pointcloud

        part_pointcloud = fun(self.pointcloud)

        if self.pointcloud2:
            part_pointcloud2 = fun(self.pointcloud2)
            self._3Dshow(part_pointcloud=part_pointcloud, part_pointcloud2=part_pointcloud2)
            return 0
        self._3Dshow(part_pointcloud=part_pointcloud)  # 可以展示部分点云数据，方便对比

    def calulate_PSNR(self, save_path=False, resolution=3000, pcerror_path=False):

        if (pcerror_path):
            resolution = resolution
        main = 'pc_error.exe'
        para1 = "-a " + self.path
        para2 = "-b " + self.path2
        para3 = "--resolution=" + str(resolution)
        para4 = "--color=1"
        para = "%s %s %s %s %s" % (main, para1, para2, para3, para4)

        r = subprocess.run(para, capture_output=True, text=True)
        if save_path:
            log = open(save_path, 'w')
            print(r.stdout, file=log)
            log.close()
        else:
            print(r.stdout)

    def custom_draw_geometry_with_key_callback(self):

        def change_background_to_black(vis):
            opt = vis.get_render_option()
            opt.background_color = np.asarray([0, 0, 0])
            return False

        def change_background_to_write(vis):
            opt = vis.get_render_option()
            opt.background_color = np.asarray([1, 1, 1])
            return False

        def destroy_window(vis):
            vis.destroy_window()
            return False

        key_to_callback = {}
        key_to_callback[ord("B")] = change_background_to_black
        key_to_callback[ord("W")] = change_background_to_write

        key_to_callback[ord("D")] = destroy_window
        open3d.visualization.draw_geometries_with_key_callbacks([self.pointcloud], key_to_callback)


class Help_math:
    def YUV2RGB(colors):
        # R,G,B (0~255)
        Y=colors[:,0] 
        U=colors[:,1] 
        V=colors[:,2]
        R= Y + ((360 * (V - 128))>>8)
        G= Y - (( ( 88 * (U - 128)  + 184 * (V - 128)) )>>8) 
        B= Y +((455 * (U - 128))>>8) 
        c = np.stack([R, G, B], axis=-1)

        return c
    
    def RGB2YUV(colors):
            def SUB_RGB2YUV(color):
                R=color[0]
                G=color[1]
                B=color[2]
                # Y(0~255),  U,V(-128~128)          
                Y = (77*R + 150*G + 29*B)>>8
                U = ((-44*R  - 87*G  + 131*B)>>8) + 128
                V = ((131*R - 110*G - 21*B)>>8) + 128
                return [Y,U,V]

            points=[]
            for color in colors:
                points.append(SUB_RGB2YUV(color))
        
            return points

    def morton_to_points(mortons):
        def deinterleave_3d(n):
            """将三维莫顿码中的位解交错"""
            n = n & 0x9249249249249249  # 保留每3位中的第1位
            a=0
            for i in range(0,64,3):
                b=(n>>i)&1;
                a=a|(b<<int(i/3))

            return a

        z=281474976836959
        # 使用 numpy 的批量操作来处理所有的莫顿码
        z = deinterleave_3d(mortons)
        y = deinterleave_3d(mortons >> 1)
        x = deinterleave_3d(mortons >> 2)
       
        # 组合结果
        
        points = np.hstack((x, y, z))

        return points


'''
    行前必须有标志">>> ",没有这行，默认不读这行
    txt终止标志： >>> end
    变量前加@，可激活auto函数，并在最后会生成一个txt文件
    NB：变量名不允许大写（大写字母是标志符号，会激活自动调用函数

    该文件会包含特殊标识符的变量@ #(有#的变量不会写入excel)
    比如：1 @morton 会自动计算坐标 #@points
          2 @残差 or @residual 会自动输出残差的分布图（折线图）
          3 @normal 会生成法向量（如果有points，自动写出含有法向量点云）(若有colors，点云会自带颜色)
          4 发现标志符&,(并且不是C&)自动激活绘图函数

    绘图部分，分为准备数据和绘制数据！
    可以绘制三类图形：星型图，直方图，折线图，密度图
    折线图又分为统计分布图，走势图
    1、准备数据：
        a： 输入准备的所有数据，如colors,colors[1],level[1]
        b:  输入分类数据，需要前加标志符号：&C,此时，会将数据分类绘制
        c:  如果数据前有标志服#，如#-1，那数据-1，不会计入统计（舍去
                         

    2、绘制数据：
        a:  在初始化函数中，选定默认绘图方式和图形类型，也可以手动调用相关函数（目前默认绘制折线图、统计分布图
        b： 绘制的图标题为已滤波：1-N（滤波条件，会将所有的；滤波数据打印到@filter.txt 文件
        c:  图横坐标为输入数据，纵坐标为radio（可自行修改
        d： 图的纵坐标默认为占比，如果想要修改，请自行修改
        f： 图的文件名称为filter1-N，不滤波的话默认为filter（亦可以手动指定
        g： 图的横坐标可以在函数头做出修改（默认全部显示

    3、希望解决的问题
        a：  调参问题（加权调参，让他自行解决，核心是加速计算和失真选择
        b：  
    '''
    
'''
    1:统计分布图
    2:走势图
    '''
drawing_method=1
'''
    1-4:星型图，直方图，折线图，密度图
    '''
drawing_shape=3

xlimit=[0,9]
ylimit=[0,1]
endeligiable=False
class Help_Statistics:

    def __init__(self,base,file_name,auto="默认",only_read=False):
         self.markers = ['o', 's', 'D', 'd', '^', 'v', '>', '<', '*', 'p', 'h', 'H', 'x', 'X', '+', '|', '_', '.']
         self.line_styles = ['-', '--', ':', '-.']

         def read_data(file_path):
            example_line=False
            with open(file_path,'r') as file:
                  for line in file:
                       if(line.find(">>> ")==0):
                           example_line=line
                           break
            if(not example_line):
                print("没有发现行标志")
                return 0
            '''
            pattern为正则表达式，如 pattern = r"#残差\s*[:：]\s*(-?[\d\s.]+)\s*region\s*[:：]\s*(-?[\d\s.]+)"
            其中#残差是变量，：是变量后的符号，\s*表示N个空格，()表示一组数据，-?表示-可有可无，[]+,表示[]内数据读N次，\d为数据类型 ,\s表示空字符

            '''
            keys = re.findall(r"(\S+)\s*[:：]", example_line)
            
            print("所有变量: ")
            print(keys)
            if only_read:
                keys=[only_read]
            self.keys=list(keys)
            self.data=dict()
            with open(file_path, 'r') as file:
                content = file.read()  
           
            if endeligiable:
                pattern = r">>> end"  # 例如 STOP_HERE 作为结束标志
                match = re.search(pattern, content)

                if match:
                    print(f"匹配到终止标志，位置: {match.start()}")
                    content = content[:match.start()]  # 只保留 STOP_HERE 之前的内容

            for key in keys:
            # 使用更高效的正则表达式查找所有匹配项
                pattern = rf">>>[^\n]*?{key}\s*[:：]\s*([-\d\s.]+)"
                matches = re.findall(pattern, content)  # 获取所有匹配的数据

                if matches:
                    digit = matches[0].split()[0]  # 获取第一个数字来判断数据类型
                
                    # 根据第一个数字是否含有小数点来判断数据类型
                    dtype = np.float64 if '.' in digit else int
                    # 将数据转换为 numpy 数组
                    self.data[key] = np.array([np.fromiter(row.split(), dtype=dtype) for row in matches])

         self.base=base

         read_data(file_name)
         self.filter=[]

         self.auto_math()
         if auto=="默认":
            self.auto_fun()
         elif auto=="上下文":
            self.auto_fun1()
            self.auto_fun2()
            self.auto_fun3()
            self.auto_fun4()
         
    def auto_math(self):
        self.Class=None
        self.Colors=None
        self.bit=None
        self.Minlevel=None
        self.Morton=None
        for key in self.keys:
            if key.find("C&")==0:
                self.Class=key
            if key.find("colors")>=0:
                self.Colors=key
            if key.find("level")>=0:
                self.Minlevel=min(self.data[key])
            if key.find("morton")>=0:
                self.Morton=key

        keys=self.keys
        for key in keys:

            if key=="@morton":
                mortons=self.data[key]
                mortons = mortons.astype(np.uint64)
                points=Help_math.morton_to_points(mortons)
                self.data["#points"]=points
                self.Points="#points"
                self.keys.append("#points")
        
    def auto_fun(self):

        keys=self.keys
        for key in keys:

            if key=="@points":
                self.generate_PointCloud(outname="idcm",Class=self.Class)

            if key.find("&")==0:
                if not self.Class==None:
                    c=np.unique(self.data[self.Class])

                    for v in c:
                        if v<=4:
                            continue
                        data=self.my_filter("data['"+self.Class+"'][:,0]=="+str(v),update=False)
                        path=self.base+"/drawing/"+self.Class
                        os.makedirs(path, exist_ok=True)
                        self.sub_draw(data[key],save_path=path+"/"+self.Class+str(v)+".png",label=self.Class+str(v))
                        #self.clear_drawing()
                    self.drawing_show()
                    #self.drawing_save("节点点数统计")
                    self.clear_drawing()
                else:
                    self.draw(key,name="init_picture")

        self.write()

    def auto_fun1(self):

        keys = self.keys
        for key in keys:

            if key == "@points":
                self.generate_PointCloud(outname="idcm", Class=self.Class)

            if key.find("&") == 0:
                if not self.Class == None:
                    levels = np.unique(self.data[self.Class])
                    gnp=np.unique(self.data["gnp"])


                    for l in levels:

                        p = np.zeros(len(gnp))
                        exp = "data['" + self.Class + "'][:,0]==" + str(l)
                        data = self.my_filter(exp, update=False)

                        rl=int(100*len(data[self.Class])/len(self.data[self.Class]))
                        if rl < 10:
                            continue
                        for i in range(len(gnp)):
                            g=gnp[i]

                            exp=["data['" + self.Class + "'][:,0]==" + str(l),"data['gnp'][:,0]==" + str(g)]
                            data = self.my_filter(exp, update=False)
                            num=len(data[self.Class])


                            if len(data[key])==0:
                                continue

                            exp.append("data['&numPoints'][:,0]<=2")
                            data = self.my_filter(exp, update=False)
                            num1 = len(data[self.Class])

                            radio=num1/num
                            p[i]=radio
                        random_marker = random.choice(self.markers)
                        random_linestyle = random.choice(self.line_styles)
                        plt.plot(p, label="level"+str(l)+"_radio:"+str(rl)+"%", marker=random_marker, linestyle=random_linestyle)
                        #plt.xlim(xlimit)
                        plt.ylim(ylimit)
                        plt.title(self.base.split("/")[-1])
                        plt.xlabel("ctx")
                        plt.ylabel("radio")
                        plt.legend(loc='upper right')
                    #self.drawing_show()

                        # self.clear_drawing()
                        #self.drawing_show()
                    os.makedirs(self.base+"/drawing", exist_ok=True)
                    self.drawing_save("drawing/"+"不同上下文IDCM节点占比")
                    self.clear_drawing()
                else:
                    self.draw(key, name="init_picture")

    def auto_fun2(self):

        keys = self.keys
        for key in keys:

            if key == "@points":
                self.generate_PointCloud(outname="idcm", Class=self.Class)

            if key.find("&") == 0:
                if not self.Class == None:
                    levels = np.unique(self.data[self.Class])
                    gnp=np.unique(self.data["gnp"])

                    print(len(gnp))
                    for l in levels:

                        p = np.zeros(len(gnp))
                        exp = "data['" + self.Class + "'][:,0]==" + str(l)
                        data = self.my_filter(exp, update=False)

                        rl=int(100*len(data[self.Class])/len(self.data[self.Class]))
                        if rl < 10:
                            continue
                        for i in range(len(gnp)):
                            g=gnp[i]

                            exp=["data['" + self.Class + "'][:,0]==" + str(l),"data['gnp'][:,0]==" + str(g)]
                            data = self.my_filter(exp, update=False)
                            num=len(data[self.Class])

                            if len(data[key])==0:
                                continue

                            exp.append("data['&numPoints'][:,0]<=1")
                            data = self.my_filter(exp, update=False)
                            num1 = len(data[self.Class])

                            radio=num1/num
                            p[i]=radio
                        random_marker = random.choice(self.markers)
                        random_linestyle = random.choice(self.line_styles)
                        plt.plot(p, label="level"+str(l)+"_radio:"+str(rl)+"%", marker=random_marker, linestyle=random_linestyle)
                        #plt.xlim(xlimit)
                        plt.ylim(ylimit)
                        plt.title(self.base.split("/")[-1])
                        plt.xlabel("ctx")
                        plt.ylabel("radio")
                        plt.legend(loc='upper right')
                    #self.drawing_show()

                        # self.clear_drawing()
                        #self.drawing_show()
                    os.makedirs(self.base+"/drawing", exist_ok=True)
                    self.drawing_save("drawing/"+"不同上下文IDCM1占比")
                    self.clear_drawing()
                else:
                    self.draw(key, name="init_picture")

    def auto_fun3(self):

        keys = self.keys
        for key in keys:

            if key == "@points":
                self.generate_PointCloud(outname="idcm", Class=self.Class)

            if key.find("&") == 0:
                if not self.Class == None:
                    levels = np.unique(self.data[self.Class])
                    gnp = np.unique(self.data["gnp"])

                    for l in levels:

                        p = np.zeros(len(gnp))
                        exp = "data['" + self.Class + "'][:,0]==" + str(l)
                        data = self.my_filter(exp, update=False)

                        rl = int(100 * len(data[self.Class]) / len(self.data[self.Class]))
                        if rl < 10:
                            continue
                        for i in range(len(gnp)):
                            g = gnp[i]

                            exp = ["data['" + self.Class + "'][:,0]==" + str(l), "data['gnp'][:,0]==" + str(g)]
                            data = self.my_filter(exp, update=False)
                            num = len(data[self.Class])

                            if len(data[key]) == 0:
                                continue

                            exp.append("data['&numPoints'][:,0]==2")
                            data = self.my_filter(exp, update=False)
                            num1 = len(data[self.Class])

                            radio = num1 / num
                            p[i] = radio
                        random_marker = random.choice(self.markers)
                        random_linestyle = random.choice(self.line_styles)
                        plt.plot(p, label="level" + str(l) + "_radio:" + str(rl) + "%", marker=random_marker,
                                 linestyle=random_linestyle)
                        # plt.xlim(xlimit)
                        plt.ylim(ylimit)
                        plt.title(self.base.split("/")[-1])
                        plt.xlabel("ctx")
                        plt.ylabel("radio")
                        plt.legend(loc='upper right')
                    # self.drawing_show()

                    # self.clear_drawing()
                    # self.drawing_show()
                    os.makedirs(self.base + "/drawing", exist_ok=True)
                    self.drawing_save("drawing/" + "不同上下文IDCM2占比")
                    self.clear_drawing()
                else:
                    self.draw(key, name="init_picture")

    def auto_fun4(self):

        keys=self.keys
        for key in keys:

            if key=="@points":
                self.generate_PointCloud(outname="idcm",Class=self.Class)

            if key.find("&")==0:
                if not self.Class==None:
                    c=np.unique(self.data[self.Class])

                    for v in c:

                        exp = "data['" + self.Class + "'][:,0]==" + str(v)
                        data = self.my_filter(exp, update=False)

                        rl = int(100 * len(data[self.Class]) / len(self.data[self.Class]))
                        if rl < 10:
                            continue
                        data=self.my_filter("data['"+self.Class+"'][:,0]=="+str(v),update=False)
                        path=self.base+"/drawing/"
                        os.makedirs(path, exist_ok=True)
                        self.sub_draw(data["gnp"],label=self.Class+str(v),t=self.base.split("/")[-1])
                        #self.clear_drawing()
                    #self.drawing_show()
                    self.drawing_save("drawing/"+"ctx占比")
                    self.clear_drawing()
                else:
                    self.draw(key,name="init_picture")

        self.write()
    #输入运算表达式
    def my_filter(self,expressions,update=True):             

        def sub_filter(data,expression):
         # 使用 eval 执行表达式
            
            filt = eval(expression)
            keys=self.keys
            rows=np.where(filt)
            for key in keys:
                data[key]= data[key][rows[0],:]

            return data

        data=copy.deepcopy(self.data)
        if isinstance(expressions,list):
            for expression in expressions:
                data=sub_filter(data,expression)
        elif isinstance(expressions,str):
            data=sub_filter(data,expressions)
        else:
            print("输入参数不合法")
            return False

        if(len(data)==0):
            print("滤波后没有数据")
            return 0

        if update:
            self.data=data
        #print("满足表达式的有"+str(len(self.data))+"个")
        return data

    def draw(self,key,num1=-1,key2=None,num2=-1,name=""):
        path = self.base + "/drawing"
        if not os.path.exists(path):
            os.mkdir(path)

        if not os.path.exists(self.base+"/drawing"):
            os.mkdir(self.base+"/drawing")

        save_path=self.base+"/drawing/"+key+"-"+name+".png"

        if key2==None:
            data2=None
        else:
            data2=self.data[key2]

        self.sub_draw(self.data[key],data2,num1=num1,num2=num2,save_path=save_path,label=key+name)

    def sub_draw(self,data,data2=None,num1=-1,num2=-1,save_path=None,label="",show=False,t=""):
        path = self.base + "/drawing"

        if not os.path.exists(path):
            os.mkdir(path)

        plt.rcParams['font.family'] = 'SimSun'  # 设置字体为宋体

        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        markers = ['o', 's', 'D', 'd', '^', 'v', '>', '<', '*', 'p', 'h', 'H', 'x', 'X', '+', '|', '_', '.']
        line_styles = ['-', '--', ':', '-.']
      # 随机选择一个 marker
        for i in range(data.shape[1]):  # 遍历每一列数据
            if num1>=0:
                i=num1

            variance = np.var(data[:,i])
            if drawing_shape==2:
                bins=max(data[:,i])-min(data[:,i])
                plt.hist(data[:, i],bins=bins,density=True, alpha=0.6, label=label+":"+str(i))

            elif drawing_shape==3:
               x = range(min(data[:,i]),max(data[:,i])+1)
               Min=min(data[:,i])
               random_marker = random.choice(markers)
               random_linestyle = random.choice(line_styles)
               p = np.zeros(len(x))
               for v in data[:,i]:
                   p[v-Min]=p[v-Min]+1
               NUM=len(data)
               for i in range(len(p)):
                   p[i]=p[i]/NUM

               plt.plot(x,p, label=label,marker=random_marker,linestyle=random_linestyle)

            elif drawing_shape==1:
                plt.scatter(data[:,num1],data2[:,num2],label=label+str(i))

            if num1>=0:
                break

        plt.xlim(xlimit)
        plt.ylim(ylimit)
        plt.title(t)
        plt.xlabel("节点点数")
        plt.ylabel("radio")
        plt.legend(loc='upper right')
        # 显示图形
        if show:
            plt.show()

        if save_path:
            plt.savefig(save_path)


    def generate_PointCloud(self,YUV=True,outname="",Class=None):    #Class为不同类，比如为level，他会分层输出点云
        f_colors=self.Colors
        path=self.base+"/pointcloud"
        if not os.path.exists(path):
                    os.mkdir(path)
        if not Class==None:
            所有类=np.unique(self.data[Class])
        else:
            所有类=["all"]

        for 类 in 所有类:
            if not Class==None:
                data=self.my_filter("data['"+Class+"']=="+str(类),update=False)
            else:
                data=self.data
            points=[]
            for k in self.keys:
                if k.find("points")>=0:
                    points=data[k]
                    break;
            if len(points)==0:
                print("无点")
                return 0
            if not Class==None:
                print(Class+str(类)+"点数:"+str(len(points)))
            else:
                print(outname+"点数:"+str(len(points)))
            point_cloud = open3d.geometry.PointCloud()
            points=np.asarray(points)

            point_cloud.points = open3d.utility.Vector3dVector(points)  # 设置点云数据
            if not f_colors==None:
                colors=data[f_colors]
                if YUV:
                    colors=Help_math.YUV2RGB(colors)
                colors=colors/255.0
            
                point_cloud.colors=open3d.utility.Vector3dVector(colors)


            for k in self.keys:
                if k.find("normal")>=0:
                    normals=data[k]/100
                    point_cloud.normals = open3d.utility.Vector3dVector(normals)
                    break

            path=self.base+"/pointcloud"

            if not os.path.exists(path):
                    os.mkdir(path)

            Help_PointCloud._write(point_cloud,path+"/"+str(类)+outname+".ply")
            
    def write(self,save_name="@data",filters=False):
       # 预处理数据，将多维数据转换为字符串
        path=self.base+'\\'+save_name+'.xlsx'
        if os.path.exists(path):
            return 0
        print("写数据相当费时间，请耐心的等待，如果输出表格存在，不会执行写数据操作")
        if filters:
            my_data=self.my_filter(filters,update=False)
        else:
             my_data=self.data

        data = dict()
        for key in self.keys:
            if key.find("#")>=0:
                continue
            size=my_data[key].shape

            if size[1]>1:
                    for i in range(size[1]):
                        data[key+"["+str(i)+"]"]=my_data[key][:,i]
            else:
                    data[key]=my_data[key][:,0]

        df = pd.DataFrame(data)

        # 将 DataFrame 写入 Excel 文件
        with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')

    def clear_drawing(self):
        plt.clf()

    def drawing_show(self):
        plt.show()

    def drawing_save(self,save_name):
        plt.legend(loc='upper right')

        plt.savefig(self.base+"/"+save_name+".png", bbox_inches="tight",dpi=1200)


    @property
    def get_pointCount(self):
        return self.data[self.keys[0]].shape[0]

    def draw_diff_ignoreValueINkey2(self,data1,data2,Value=None,label=""):   #剔除1中的值
            if data2.ndim==1:
                data2=data2.reshape(-1,1)
            if data1.ndim==1:
                data1=data1.reshape(-1,1)
            if not Value==None:
                filt = data1[:,0]!=Value
                rows=np.where(filt)
                data1=data1[rows[0],:]
                data2=data2[rows[0],:]
            self.sub_draw(data1-data2,label=label)     
    
    def _3D_show(self,width=6,show_key=False):
        def sub_Show(width,morton):
            
            indice = np.where(self.data[self.Morton] == morton)[0]
            
            center=self.data[self.Points][indice,:]
            box_min=center-width
            box_max=center+width
            filtered_indices = np.where(np.all(self.data[self.Points] > box_min , axis=1))
            data= self.data[self.Points][filtered_indices]
            mortons=self.data[self.Morton][filtered_indices]
            YUV=self.data[self.Colors][filtered_indices]
            if show_key:
                show=self.data[show_key][filtered_indices]
            else:
                show=YUV

            filtered_indices = np.where(np.all(data <= box_max , axis=1))
            points= data[filtered_indices]
            mortons=mortons[filtered_indices]
            mortons_idx = np.where(np.logical_and(((morton >> 3) << 3) <= mortons,mortons <= (morton | 0x7)))[0]

            YUV=YUV[filtered_indices]
            show=show[filtered_indices]
            rgb=Help_math.YUV2RGB(YUV)
            x, y, z = points[:, 0], points[:, 1], points[:, 2]
            indice = np.where(np.all(points == center,axis=1))
            rgb[mortons_idx,:]=[1,1,0]
            rgb[indice,:]=[0,0,1]
            # 创建 3D 散点图，带有坐标标注
            fig = go.Figure()

            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers+text',  # 同时显示点和文本
                marker=dict(size=5, color=rgb),  # 设置点的大小和颜色
                text=[f'({show[i,0]:.1f}' for i in range(len(x))],  # 标注坐标
                textposition="top center"  # 设置文本位置
            ))

            fig.update_layout(
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z'
                ),
                title=str(points[indice,:])
            )

            fig.show()

        while 1:
            morton = int(input("请输入一个点的莫顿码: "))
            if morton==-1:
                break
            sub_Show(width,morton)

#调参模块
class Help_Tuning_parameters:
    def __init__(self,ply_path,cfg_path):
        print("路径请在初始化函数配置")
        self.base=r'C:\Users\31046\Desktop\tmc13-v29-集成代码结果\编解码匹配验证'                #工作路径
        self.test=r'C:\Users\31046\Desktop\tmc13-v29-集成代码结果\tmc13-v29-rc1\encoder'      #测试代码
        self.ply_path=ply_path
        self.cfg_path=cfg_path
        self.result=self.base+"\\result.txt"
        
    def run(self):
        if os.path.exists(self.result):
            os.remove(self.result)
       
        for new_parent_weight in range(15,40,2):
            for new_child_weight in range(40,70,2):
                self.Tuning_parameters(new_parent_weight,new_child_weight,new_num = 4)
                self.generate_PointCloud()
                self.calulate_PSNR()
     
    def generate_PointCloud(self):
            main=self.test+"\\build\\tmc3\\Release\\"+"tmc3.exe"
           
            para_encfg="-c "+self.cfg_path
      
            para2="--uncompressedDataPath="+self.ply_path
            para3="--compressedStreamPath="+self.base+"\\compress.bin"
            para4="--reconstructedDataPath="+self.base+"\\recon.ply"
            para="%s %s %s %s %s"%(main,para_encfg,para2,para3,para4)
          
            r=subprocess.run(para,capture_output=True,text=True)
            log=open(self.result,'a')
            print(r.stdout,file=log)
            log.close()
            if not os.path.exists(self.base+"\\recon.ply"):
                print("编码端重建失败")
                error_out="%s %s %s %s"%(para_encfg,para2,"-compressedStreamPath=compress.bin","-reconstructedDataPath=recon_encoder.ply")
                print("运行配置为: "+error_out)
            else:
                print("点云生成成功")
                    
    def Tuning_parameters(self,new_parent_weight = 55,new_child_weight = 65,new_num = 6):
        # 更新参数
        cfg_path=r'C:\Users\31046\Desktop\tmc13-v29-集成代码结果\tmc13-v29-rc1\encoder\build\test\cfg.txt'
        # 读取文件
        with open(cfg_path, 'r') as file:
            lines = file.readlines()

        updated_lines = []
        for line in lines:
            if line.startswith('parent_weight'):
                updated_lines.append(f'parent_weight:{new_parent_weight}\n')
            elif line.startswith('child_weight'):
                updated_lines.append(f'child_weight:{new_child_weight}\n')
            elif line.startswith('num'):
                updated_lines.append(f'num:{new_num}\n')
            else:
                updated_lines.append(line)

        # 将更新后的内容写回文件
        with open(cfg_path, 'w') as file:
            file.writelines(updated_lines)
        with open(self.result, 'a') as file:
            file.writelines(updated_lines)

        print("文件参数已更新。")

    def calulate_PSNR(self,resolution=3000,pcerror_path=False):
         
         if(pcerror_path):
             resolution=resolution
         main='pc_error.exe'
         para1="-a "+self.ply_path
         para2="-b "+self.base+"\\recon.ply"
         para3="--resolution="+str(resolution)
         para4="--color=1"
         para="%s %s %s %s %s"%(main,para1,para2,para3,para4)

         r=subprocess.run(para,capture_output=True,text=True)
         log=open(self.result,'a')
         print(r.stdout,file=log)
         log.close()
         os.remove(self.base+"\\recon.ply")
         print("PSNR计算完毕")

    def read_result(self):
        self.data=dict()
        with open(self.result, 'r') as file:
                content = file.read()  
        patterns=["parent_weight:([\d]+)","child_weight:([\d]+)","num:([\d]+)","colors bitstream size ([\d]+)",r'c\[0\],PSNRF\s*:\s*([\d\.]+)']
        for p in patterns:
              matches = re.findall(p, content)  # 获取所有匹配的数据
              if p=="colors bitstream size ([\d]+)":
   
                  self.data["even"] = np.array([np.fromiter(matches[i].split(), dtype=int) for i in range(0,len(matches),2)])
                  self.data["odd"] = np.array([np.fromiter(matches[i].split(), dtype=int) for i in range(1,len(matches),2)])
                  continue
              if matches:
                    digit = matches[0].split()[0]  # 获取第一个数字来判断数据类型
                
                    # 根据第一个数字是否含有小数点来判断数据类型
                    dtype = np.float64 if '.' in digit else int
                    # 将数据转换为 numpy 数组
                    self.data[p] = np.array([np.fromiter(row.split(), dtype=dtype) for row in matches])

        path=self.base+'\\'+'result.xlsx'

        my_data=self.data
        data = dict()
        for key in self.data.keys():
                size=my_data[key].shape
                if size[1]>1:
                    for i in range(size[1]):
                        data[key+"["+str(i)+"]"]=my_data[key][:,i]
                else:
                    data[key]=my_data[key][:,0]

        df = pd.DataFrame(data)

        # 将 DataFrame 写入 Excel 文件
        with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')


        
         