import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import copy
import heapq
import distinctipy
from scipy.spatial import ConvexHull
import cv2
import matplotlib.image as mpimg
from PIL import Image, ImageDraw, ImageFont
import time
import tempfile
import textwrap

class VoronoiTreemap:
    def __init__(self, save_path, data_name = "data", radius = 600, hierarchy_index: int = 0, hierarchies_num: int = 2, higher_hierarchy = [], valid_points = None, maptree_info = {}):
        self.data_name = data_name
        if hierarchy_index != 0:
            for i in range(hierarchy_index):
                data_name = data_name + '_' + str(higher_hierarchy[hierarchy_index - 1])
        data_path = data_name + '.csv'
        
        self.radius = radius
        self.save_path = save_path
        
        self.generator_num, self.init_data = self._read_data(data_path)
        print("n: ", self.generator_num)
        
        self.valid_points = valid_points
        self.desired_areas = []
        self.generator_info = {}
        self.maptree_info = maptree_info
        self.generator_area = {}
        self.areas = []
        self.higher_hierarchy = higher_hierarchy
        self.hierarchy_index = hierarchy_index
        self.hierarchies_num = hierarchies_num
    
    def _read_data(self, data_path):
        """
        从.csv文件中读取数据（保存在self.init_data）
        """
        init_data = []
        with open(data_path, "r", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            for i, row in enumerate(reader):
                init_data.append(row)
                n = i + 1
        return n, init_data
    
    def voronoi_treemap_processor(self, generator_info: dict = {}, boundaries: list = []) -> dict:
        """
        递归生成treemap（DFS）
        生成的信息由 generator_info 和 maptree_info 储存，在遍历过程中实时更新：
        generator_info: dict，key：tuple，和node一一对应；value：由node key生成的所有generator的信息，包括generator在图里的坐标，label文本，area值
        maptree_info: dict，key：像素坐标(y, x)，value：tuple，为像素点(y, x)属于的叶子node生成的generator的信息
        """
        print("index: ", self.hierarchy_index)
        print("self.higher_hierarchy: ", self.higher_hierarchy)
        if self.hierarchy_index == 0:
            self._init_gragh()
        
        print("before submap_generator")
        res = self._childmap_generator()
        if res is None:
            return
        generator_info[tuple(self.higher_hierarchy)] = self.generator_info
        print("after submap_generator")
        
        # 如果不是叶子节点，递归得到子图
        if self.hierarchy_index != self.hierarchies_num - 1:
            for i in range(self.generator_num):
                next_higher_hierarchy = copy.deepcopy(self.higher_hierarchy)
                next_higher_hierarchy.append(i)
                print("next_higher_hierarchy: ", next_higher_hierarchy)
                child_res = None
                while child_res is None:
                    child_map = VoronoiTreemap( data_name = self.data_name, 
                                                hierarchy_index = self.hierarchy_index + 1, 
                                                hierarchies_num = self.hierarchies_num,
                                                higher_hierarchy = next_higher_hierarchy,
                                                valid_points = self.generator_area.get(i), 
                                                maptree_info = copy.deepcopy(self.maptree_info),
                                                save_path = self.save_path)
                    child_res = child_map.voronoi_treemap_processor(generator_info=generator_info, boundaries=boundaries)
                    if child_res is not None:
                        self.maptree_info = child_res
                
                if self.hierarchy_index != 0:
                    return self.maptree_info
                
            print("generator_info: ", generator_info)
            # 在根节点，生成 voronoi treemap 图片
            self._image_generator(generator_info = generator_info)
            return self.maptree_info
        
        # 如果是叶子节点，返回maptree_info
        else:
            return self.maptree_info
    
    def _image_generator(self, generator_info):
        """
        生成voronoi treemap图片
        1. 每个叶子node生成的generator对应不同的颜色，绘制 img_for_edge.png
        2. 利用形态变换（膨胀、腐蚀），提取 img_for_edge.png 中的边缘，保存为二值图像edge.png
        3. 根节点node(,)生成的generator对应不同的颜色，即根据hierarchy  = 0的generator的分类，划分颜色区域，绘制
        4. 利用edge.png为3中绘制的图像添加边界，并利用generator_info为每个叶子node生成的generator添加label文本
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            img4edge_path = f"{temp_dir}/img_for_edge.png"
            
            generator_num = 0
            for value in generator_info.values():
                generator_num += len(value)
            
            self._draw_for_edge(save_path = img4edge_path, generator_num = generator_num)
            
            edge_img_path = f"{temp_dir}/edge.png"
            edge_result = self._edge_generator(save_path = edge_img_path, img_path = img4edge_path)
            self._add_edge_and_label(generator_info = generator_info, edges = edge_result)
    
    def _edge_generator(self, save_path = "/home/hanqing/voronoi_treemap/edge.png", img_path = "/home/hanqing/voronoi_treemap/for_edge.png", kernel_size = 4):
        """
        利用膨胀、腐蚀，提取img_path的边缘，保存在save_path
        """
        image = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        # 生成kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size, kernel_size))
        # 膨胀
        dilate_img = cv2.dilate(image, kernel)
        # 腐蚀
        erode_img = cv2.erode(image, kernel) 
        # 做差
        absdiff_img = cv2.absdiff(dilate_img,erode_img)
        # 二值化
        _, threshold_img = cv2.threshold(absdiff_img, 0, 255, cv2.THRESH_BINARY)
        # 取反，边缘为黑色
        result = cv2.bitwise_not(threshold_img)
        
        cv2.imwrite(save_path, result)

        return result
    
    def _add_edge_and_label(self, generator_info, edges, ignored_size = 1e-4):
        """
        根据最高层generator的分类，划分颜色区域
        并添加边界和label
        如果叶子generator的面积小于ignored_size，则不添加label
        """
        n = self.generator_num
        
        # 生成generator_num个颜色
        # colors = VoronoiTreemap.color_generator(n)
        # colors = [[212, 186, 176, 255], [127, 134, 123, 255], [193, 171, 173, 255], [199, 199, 187, 255], [239, 237, 231, 255]]
        colors = [[254,197,126,255], [206,209,162,255], [255,224,144,255], [163,196,177,255], [154,145,164,255], [118,171,202,255], [199,163,141,255]]
        # colors = [[255, 204, 225,255], [255, 240, 179,255], [238, 255, 194,255], [199, 237, 236,255], [207, 200, 194,255], [150, 174, 227,255], [194, 165, 232,255]]
        
        # 根据最高层generator的分类，划分颜色区域，生成图片
        for i in range(n):
            self._draw_subgragh(points=self.generator_area[i], color=colors[i])
        
        # 添加边界
        image_data = self.gragh
        for y in range(image_data.shape[0]):
            for x in range(image_data.shape[1]):
                if edges[y, x] == 0:
                    image_data[y, x][3] = 0
        
        image = Image.fromarray(image_data, 'RGBA')
        # image.show()
        image.save(self.save_path, format='PNG')
        
        # 添加label
        # 防止label超出图片边界，添加透明边框
        border_size = 1000
        VoronoiTreemap.add_transparent_border(self.save_path, self.save_path, border_size)
        
        image = Image.open(self.save_path).convert("RGBA")  
        draw = ImageDraw.Draw(image)
        image_data = np.array(image)
        
        label_boxs = []
        for gragh_node in generator_info:
            hierarchy_depth = len(gragh_node) + 1
            
            label_color = int(0.1 * 255)    # int((0.1 + 0.6 * (1 - hierarchy_depth / self.hierarchies_num)) * 255)
            
            info = generator_info[gragh_node]
            for generator in info:
                i_generator_info = info.get(generator)
                if i_generator_info.get("area") < ignored_size:
                    continue
                
                if hierarchy_depth == 1:
                    label = i_generator_info.get("label")
                    font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "font", "GEORGIAB.TTF")
                    font_size = 35
                    font = ImageFont.truetype(font_path, font_size)
                    
                    bbox = draw.textbbox((0, 0), label, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    area_center_x = i_generator_info.get("x")
                    area_center_y = i_generator_info.get("y")
                    area_center_x = area_center_x + border_size
                    area_center_y = area_center_y + border_size
                    
                    gragh_center_x = self.radius + border_size
                    gragh_center_y = self.radius + border_size
                    
                    start_point_x, start_point_y, end_point_x, end_point_y, label_box = self._line_processer(area_center_x, area_center_y, gragh_center_x, gragh_center_y, text_width, text_height, image_data, border_size, label_boxs)
                    label_boxs.append(label_box)
                    
                    # 添加label文本
                    if start_point_x > end_point_x:
                        if start_point_y > end_point_y:
                            draw.text((end_point_x - 5, end_point_y - 5), label, font=font, anchor="rb", fill=(label_color, label_color, label_color, 255))
                        else:
                            draw.text((end_point_x - 5, end_point_y + 5), label, font=font, anchor="rt", fill=(label_color, label_color, label_color, 255))
                    else:
                        if start_point_y > end_point_y:
                            draw.text((end_point_x + 5, end_point_y - 5), label, font=font, anchor="lb", fill=(label_color, label_color, label_color, 255))
                        else:
                            draw.text((end_point_x + 5, end_point_y + 5), label, font=font, anchor="lt", fill=(label_color, label_color, label_color, 255))
                    
                    # 添加label连线，连线所在直线经过area_center
                    draw.line((start_point_x, start_point_y, end_point_x, end_point_y), fill=(label_color, label_color, label_color, 255), width=2)
                    
                if hierarchy_depth == self.hierarchies_num:
                    font_times = 7
                    wrap_widths = [1000, 15, 10, 7, 5]
                    wrap_times = len(wrap_widths)
                    
                    init_font_size = 40
                    
                    label_done = False
                    for i in range(font_times):
                        if label_done:
                            break
                        origin_label = i_generator_info.get("label")
                        
                        ifont_size = init_font_size - 4 * i
                        font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "font", "CALIFR.TTF")
                        font = ImageFont.truetype(font_path, ifont_size)
                        
                        for j in range(wrap_times):
                            if label_done:
                                break
                            
                            label = textwrap.fill(origin_label, width=wrap_widths[j], break_long_words=False)

                            # 获取文本大小
                            bbox = draw.textbbox((0, 0), label, font=font)
                            text_width = bbox[2] - bbox[0]
                            text_height = bbox[3] - bbox[1]

                            # 计算文本中心坐标
                            center_x = i_generator_info.get("x")
                            center_y = i_generator_info.get("y")
                            text_x = center_x
                            text_y = center_y

                            text_x = text_x + border_size
                            text_y = text_y + border_size

                            enough_space = True
                            for y in range(int(text_y - text_height / 2), int(text_y + text_height / 2)):
                                for x in range(int(text_x - text_width / 2), int(text_x + text_width / 2)):
                                    if image_data[y, x][3] == 0:
                                        enough_space = False
                                        break
                                    
                            if enough_space:
                                draw.text((text_x, text_y), label, font=font, anchor="mm", fill=(label_color, label_color, label_color, 255))
                                label_done = True

        # image.show()
        alpha = image.split()[3]
        bbox = alpha.getbbox()
        if bbox:
            margin = 10
            left = max(bbox[0] - margin, 0)
            upper = max(bbox[1] - margin, 0)
            right = min(bbox[2] + margin, image.width)
            lower = min(bbox[3] + margin, image.height)
            image = image.crop((left, upper, right, lower))
        image.save(self.save_path, format='PNG')
    
    def _line_processer(self, area_center_x, area_center_y, gragh_center_x, gragh_center_y, text_width, text_height, image_data, border_size, label_boxs):
        """
        找到单个label连线的起点和终点，满足不同label没有overlap
        """
        k = (area_center_y - gragh_center_y) / (area_center_x - gragh_center_x)
        m, n, r = self.radius + border_size, self.radius + border_size, self.radius
        b = area_center_y - k * area_center_x
        
        done = False
        while not done:
            # y = kx + b, (x - m)^2 + (kx + b - n)^2 = r^2
            A = 1 + k**2
            B = 2 * (k * (b - n) - m)
            C = m**2 + (b - n)**2 - r**2
            
            # 解方程
            delta = B**2 - 4 * A * C
            if delta < 0:
                return
            
            x1 = (-B + np.sqrt(delta)) / (2 * A)
            y1 = k * x1 + b
            x2 = (-B - np.sqrt(delta)) / (2 * A)
            y2 = k * x2 + b
            start_point_x, start_point_y = (x1, y1) if abs(x1 - area_center_x) < abs(x2 - area_center_x) else (x2, y2)
            
            k_line = abs((area_center_y - start_point_y) / (area_center_x - start_point_x))
            end_point_x = start_point_x + np.sign(start_point_x - area_center_x) * 60 / np.sqrt(1 + k_line ** 2)
            end_point_y = start_point_y + np.sign(start_point_y - area_center_y) * 60 / np.sqrt(1 + k_line ** 2) * k_line
            
            overlap = False
            if len(label_boxs) == 0:
                done = True
                if start_point_x > end_point_x:
                    if start_point_y > end_point_y:
                        m_min_x, m_min_y, m_max_x, m_max_y = end_point_x - text_width, end_point_y - text_height, end_point_x, end_point_y
                    else:
                        m_min_x, m_min_y, m_max_x, m_max_y = end_point_x - text_width, end_point_y, end_point_x, end_point_y + text_height
                else:
                    if start_point_y > end_point_y:
                        m_min_x, m_min_y, m_max_x, m_max_y = end_point_x, end_point_y - text_height, end_point_x + text_width, end_point_y
                    else:
                        m_min_x, m_min_y, m_max_x, m_max_y = end_point_x, end_point_y, end_point_x + text_width, end_point_y + text_height
            else:
                for i in range(len(label_boxs)):
                    min_x, min_y, max_x, max_y = label_boxs[i]
                    if start_point_x > end_point_x:
                        if start_point_y > end_point_y:
                            m_min_x, m_min_y, m_max_x, m_max_y = end_point_x - text_width, end_point_y - text_height, end_point_x, end_point_y
                        else:
                            m_min_x, m_min_y, m_max_x, m_max_y = end_point_x - text_width, end_point_y, end_point_x, end_point_y + text_height
                    else:
                        if start_point_y > end_point_y:
                            m_min_x, m_min_y, m_max_x, m_max_y = end_point_x, end_point_y - text_height, end_point_x + text_width, end_point_y
                        else:
                            m_min_x, m_min_y, m_max_x, m_max_y = end_point_x, end_point_y, end_point_x + text_width, end_point_y + text_height

                    if m_min_x > max_x or m_max_x < min_x or m_min_y > max_y or m_max_y < min_y:
                        continue
                    else:
                        print("overlap")
                        overlap = True
                        theta = np.arctan(k)
                        theta = theta + np.pi / 12  # 每次转15°
                        k = np.tan(theta)
                        b = area_center_y - k * area_center_x
                        break
            
            if not overlap:
                done = True
                # Bresenham算法
                x0, y0 = int(area_center_x), int(area_center_y)
                x1, y1 = int(start_point_x), int(start_point_y)

                dx, dy = abs(x1 - x0), abs(y1 - y0)
                sx = 1 if x0 < x1 else -1
                sy = 1 if y0 < y1 else -1
                err = dx - dy
    
                border_start_point = None
                border_area = False
                while True:
                    if x0 == x1 and y0 == y1:
                        break
                    
                    point = (y0, x0)
                    if image_data[point][3] == 0:
                        if self.maptree_info.get((point[0] - border_size, point[1] - border_size)) is None:
                            break
                        if not border_area:
                            border_start_point = point
                            border_area = True
                    else:
                        border_area = False
                        if self.maptree_info.get((point[0] - border_size, point[1] - border_size))[0] != self.maptree_info.get((int(area_center_y - border_size), int(area_center_x - border_size)))[0]:
                            start_point_x = border_start_point[1]
                            start_point_y = border_start_point[0]
                            break
                        
                    e2 = 2 * err
                    if e2 > -dy:
                        err -= dy
                        x0 += sx
                    if e2 < dx:
                        err += dx
                        y0 += sy
                        
        return start_point_x, start_point_y, end_point_x, end_point_y, (m_min_x, m_min_y, m_max_x, m_max_y)
    
    @staticmethod
    def add_transparent_border(image_path, output_path, border_size):
        """
        为图片添加透明边框
        """
        image = Image.open(image_path).convert("RGBA")
        width, height = image.size  

        new_width = width + 2 * border_size
        new_height = height + 2 * border_size
        new_image = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))  

        # 将原图粘贴到新画布的中心
        new_image.paste(image, (border_size, border_size), image)
        new_image.save(output_path, format="PNG")
    
    @staticmethod
    def color_generator(n: int) -> list:
        """
        生成n个不同的易于区分的颜色
        """
        colors = distinctipy.get_colors(n)
        colors = [[int(c * 255) for c in color] for color in colors]
        return colors
    
    def _childmap_generator(self, error = 0.05):
        """
        生成单层次voronoi图，算法和文献中的PW算法一致：Balzer M, Deussen O. Voronoi treemaps[C]
        基本思想：生成n个generators，有效区域内的任意一个点划分进距离最近的generator的area，
        迭代使得每个generator的面积接近desired area
        error: 误差阈值，当每个generator的area和desired area的差值小于error时，停止迭代
        """
        # 计算desired area
        self._compute_area_desired_ratio()
        self._init_points()
        
        n = self.generator_num
        stable = False
        time = 0
        
        while not stable:
            # 防止无限循环
            if time > 200:
                return
            print("time: ", time)
            time = time + 1
            
            stable = True
            
            print("before _compute_voronoi_tessellation")
            # 将有效区域内的任意一个点划分进距离最近的generator的area
            self._compute_voronoi_tessellation()
            print("after _compute_voronoi_tessellation")
            
            # 计算每个generator的area
            self._compute_area_ratio()
            
            error_list = []
            for i in range(n):
                error_list.append(abs(self.desired_areas[i] - self.areas[i]))
                if abs(self.desired_areas[i] - self.areas[i]) > error:
                    stable = False
                    # break
            # print("error_list: ", error_list)
            # print("aver_error: ", sum(error_list)/n)
            
            # 调整每个generator的weight
            for i in range(n):
                self.generator_info[i]["weight"] = self._adjust_weight(self.generator_info[i].get("weight"), 
                                                                        self.areas[i], self.desired_areas[i])
            # 将generator移动到area的质心
            self._move_generators()
        
        # 循环结束后更新generator_info
        for i in range(self.generator_num):
            self.generator_info[i]["area"] = self.areas[i]
            
        return self.maptree_info
    
    def _draw_for_edge(self, generator_num: int, save_path = "/home/hanqing/voronoi_treemap/for_edge.png"):
        """
        生成每个叶子generator为不同颜色的treemap，用于边缘提取
        """
        # generator_info里面有多少个generator，就生成多少个颜色。colors的key为tuple（可选）
        colors = {}
        colors_list = [(255 / generator_num * i, 255 / generator_num * i, 255 / generator_num * i, 255) for i in range(generator_num)]  # VoronoiTreemap.color_generator(generator_num)
        # np.random.shuffle(colors_list)
        color_num = 0
        
        for i, ipoint in enumerate(self.valid_points):
            info_list = self.maptree_info.get(ipoint)
            
            for j in range(self.hierarchies_num - 1):
                if colors.get(info_list[j]) is None:
                    colors[info_list[j]] = {}
                color = colors[info_list[j]]
                
            if color.get(info_list[self.hierarchies_num - 1]) is None:
                color[info_list[self.hierarchies_num - 1]] = colors_list[color_num]  # np.random.rand(1, 3)[0] * 255
                color_num = color_num + 1
                
            icolor = color[info_list[self.hierarchies_num - 1]]
            self.gragh[ipoint][0] = icolor[0]
            self.gragh[ipoint][1] = icolor[1]
            self.gragh[ipoint][2] = icolor[2]
        
        image = Image.fromarray(self.gragh, 'RGBA')
        image.save(save_path, format='PNG')
    
    def _init_gragh(self):
        """
        根据半径创建图片，得到self.gragh_array
        背景透明，黑色圆形
        """
        self.valid_points = []
        radius = self.radius
        width = height = 2 * radius
        
        image_data = np.zeros((width, height, 4), dtype=np.uint8)
        
        center_x, center_y = width / 2, height / 2  # 圆心位置
        
        for x in range(width):
            for y in range(height):
                distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

                if distance <= radius:
                    image_data[y, x] = [0, 0, 0, 255]
                    self.valid_points.append((y, x))
        
        self.gragh = image_data
        
        image = Image.fromarray(image_data, 'RGBA')
        image.save(self.save_path, format='PNG')
    
    def _compute_area_desired_ratio(self):
        """
        根据init_data, 得到self.desired_areas: list
        """
        gdp_list = []
        
        for i in range(self.generator_num):
            gdp_list.append(float(self.init_data[i].get("Value")))
        total_gdp = sum(gdp_list)
        
        for i in range(self.generator_num):
            self.desired_areas.append(gdp_list[i] / total_gdp)
            
        if sum(self.desired_areas) != 1:
            self.desired_areas[self.generator_num - 1] = 0
            self.desired_areas[self.generator_num - 1] = 1 - sum(self.desired_areas)
        print("desired_areas: ", self.desired_areas)
    
    def _init_points(self):
        """
        随机初始化generators, weight = 1
        得到self.generator_info: dict, 记录：第i个generator的坐标，weight
        """
        np.random.seed(None)
        n = self.generator_num
        r = self.radius
        for i in range(n):
            x = np.random.randint(0, r)
            y = np.random.randint(0, r)
            # TODO
            print("label: ", self.init_data[i].get("Label"))
            print(self.init_data[i])
            self.generator_info[i] = {"x": x, "y": y, "weight": 1, "label": self.init_data[i].get("Label"), "area": self.desired_areas[i]}
            if (y, x) not in self.valid_points:
                i = i - 1
                continue
        # print("generator_info: ", self.generator_info)
    
    def _compute_voronoi_tessellation(self):
        """
        根据generators和weights, 
        得到self.maptree_info: dict，记录每个点属于哪个generator
        得到self.generator_area: dict{list}，记录属于第i个generator的点集list
        """
        self.generator_area.clear()
        n = self.generator_num
        for i, ipoint in enumerate(self.valid_points):
            closest_generator = 0   # 最近的generator
            min_distance = float('inf')
            y = ipoint[0]
            x = ipoint[1]
            
            for j in range(n):
                generator_x = self.generator_info[j].get("x")
                generator_y = self.generator_info[j].get("y")
                weight = self.generator_info[j].get("weight")
                if self.generator_area.get(j) is None:
                    self.generator_area[j] = []
                    self.generator_area[j].append((generator_y, generator_x))
                # 用weight计算PW距离
                distance = (x - generator_x) ** 2 + (y - generator_y) ** 2 - weight
                
                if distance < min_distance:
                    min_distance = distance
                    closest_generator = j
            
            self.maptree_info[(y, x)] = tuple(self.higher_hierarchy) + (closest_generator,)
            self.generator_area[closest_generator].append((y, x))
    
    def _compute_area_ratio(self):
        """
        根据generator_area, 得到self.areas: list
        """
        self.areas = []
        total_size = len(self.valid_points)
        n = self.generator_num
        for i in range(n):
            area_size = len(self.generator_area[i])
            self.areas.append(area_size / total_size)
        if sum(self.areas) != 1:
            self.areas[n - 1] = 0
            self.areas[n - 1] = 1 - sum(self.areas)
    
    def _adjust_weight(self, weight, area, area_desired) -> float:
        """
        根据area和area_desired调整generator的weight
        """
        weight = weight * (1 + (area_desired - area) / area_desired)
        if weight < 1:
            weight = 1
        return weight
    
    def _move_generators(self):
        """
        把generator移动到area的质心
        """
        n = self.generator_num
        for i in range(n):
            x_sum = 0
            y_sum = 0
            points_in_generator = self.generator_area.get(i)
            
            for j, jpoint in enumerate(points_in_generator):
                x_sum = x_sum + jpoint[1]
                y_sum = y_sum + jpoint[0]
            
            self.generator_info[i]["x"] = x_sum / len(self.generator_area[i])
            self.generator_info[i]["y"] = y_sum / len(self.generator_area[i])
    
    def _draw_subgragh(self, points: list, color):
        """
        将points中所有点的颜色设为color
        """
        for i, ipoint in enumerate(points):
            y = int(ipoint[0])
            x = int(ipoint[1])
            self.gragh[y, x][0] = color[0]
            self.gragh[y, x][1] = color[1]
            self.gragh[y, x][2] = color[2]


current_dir = os.path.dirname(os.path.abspath(__file__))
test = VoronoiTreemap(  save_path=os.path.join(current_dir, "result.png"),
                        data_name=os.path.join(current_dir, "data1", "data"))

start_time = time.time()
test.voronoi_treemap_processor()
end_time = time.time()
print(f"running time: {end_time - start_time:.6f} 秒")
