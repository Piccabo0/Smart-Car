import numpy as np
import cv2
"""
把位置和颜色对应起来
"""

# -------------函数定义----------------------
# 选择颜色
def choose_color(color1,color2):
    color_lower =[]
    color_upper =[]
    if color1 == 'red' :# 红色区间170-180
        color_lower.append(np.array([170, 100, 130]))
        color_upper.append(np.array([180, 255, 255]))
    elif color1 == 'green':# 绿色区间55-77
        color_lower.append(np.array([55, 43, 46]))
        color_upper.append(np.array([77, 255, 255]))
    elif color1 == 'yellow':# 黄色区间20-30
        color_lower.append(np.array([21, 43, 46]))
        color_upper.append(np.array([30, 255, 255]))
    elif color1 == 'blue' :# 蓝色区间90-100
        color_lower.append(np.array([90, 43, 46]))
        color_upper.append(np.array([100, 255, 255]))
    elif color1 == 'orange':# 橙色区间0-20
        color_lower.append(np.array([0, 100, 46]))
        color_upper.append(np.array([20, 200, 255]))
    
    if color2 == 'red' :# 红色区间170-180
        color_lower.append(np.array([170, 100, 130]))
        color_upper.append(np.array([180, 255, 255]))
    elif color2 == 'green':# 绿色区间55-77
        color_lower.append(np.array([55, 43, 46]))
        color_upper.append(np.array([77, 255, 255]))
    elif color2 == 'yellow':# 黄色区间20-30
        color_lower.append(np.array([21, 43, 46]))
        color_upper.append(np.array([30, 255, 255]))
    elif color2 == 'blue' :# 蓝色区间90-100
        color_lower.append(np.array([90, 43, 46]))
        color_upper.append(np.array([100, 255, 255]))
    elif color2 == 'orange':# 橙色区间0-20
        color_lower.append(np.array([0, 100, 46]))
        color_upper.append(np.array([20, 200, 255]))

    return color_lower,color_upper

# 底层处理，提取mask进行位与运算，把目标颜色区域粗略的扣出
def color_cut(frame,color_lower,color_upper):
    
    # change to hsv model
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # get mask 利用inRange()函数和HSV模型中蓝色范围的上下界获取mask，mask中原视频中的蓝色部分会被弄成白色，其他部分黑色。
    mask = cv2.inRange(hsv, color_lower, color_upper)
    # detect blue 将mask于原视频帧进行按位与操作，则会把mask中的白色用真实的图像替换：
    res = cv2.bitwise_and(frame, frame, mask=mask)
    #cv2.imshow('Result', res)

    return mask,res

# 图形学处理，返回目标区域对应的img
def graphics_process(th):
    # 开运算去噪声
    kernel1 = np.ones((10,10),np.uint8)
    img1 = cv2.erode(th, kernel1, iterations=1)
    img1 = cv2.dilate(img1,kernel1,iterations = 1)
    # 开运算去噪声
    kernel2 = np.ones((5,5),np.uint8)
    img2 = cv2.erode(img1, kernel2, iterations=1)
    img2 = cv2.dilate(img2,kernel2,iterations = 1)

    #闭运算弥补一些连接部分
    kernel3 = np.ones((20,20),np.uint8)
    img3 = cv2.dilate(img2,kernel3,iterations = 1)
    img3 = cv2.erode(img3, kernel3, iterations=1)

    return img3

# 把识别目标的框画出来，按面积阈值去除非目标区域
def draw_box(frame,cnts):
    box_num = len(cnts)# 识别到了几个框
    box_res = []#将box的四个坐标封装储存
    box_res_num = 0# 花了几个框
    box_x =[] #框的最小x坐标
    print('Detect Box Num = ',box_num) 
    for i in range(box_num):
        rect = cv2.minAreaRect(cnts[i])# 最小外接矩形的rect = tuple((x, y), (w, h), angle)，最后一维是旋转角度
        if rect[1][0]*rect[1][1]>1500:
            box_res_num = box_res_num + 1 
            box = cv2.boxPoints(rect)# 获取矩形四个顶点，浮点型
            box_res.append(box)
            cv2.drawContours(frame, [np.int0(box)], -1, (0, 255, 255), 2)# int0()取整
            box_x.append((box_res[i][3][0]+box_res[i][2][0]+box_res[i][1][0]+box_res[i][0][0])/4) # 框最小的横坐标
    # cv2.imshow('1',frame)
    # cv2.waitKey(0)
    print("Draw Box Num=",box_res_num)
    return box_res,box_res_num,box_x

# 对目标点读取其颜色
def get_color(hue_value):
    color='UNKNOW'
    if hue_value<20:
        color = 'ORANGE'
    elif hue_value<33:
        color = "YELLOW"
    elif hue_value<78:
        color = "GREEN"
    elif hue_value<131:
        color = "BLUE"
    elif hue_value>170 and hue_value<180:
        color = 'RED'
    else:
        color = 'UNKNOW'
    return color

# 把框中心点的颜色识别出来，作为框选区域的颜色，打印在图片上
def detect_box_color(frame,box_res,box_res_num):
    color_res = []
    hsv_frame=cv2.cvtColor(src=frame,code=cv2.COLOR_BGR2HSV) 
    for i in range(box_res_num):# 从大到小画框
        # 纵坐标最小的是top_point，纵坐标最大的是bottom_point， 
        # 横坐标最小的是left_point，横坐标最大的是right_point
        # np.array([[top_point_x, top_point_y], 
        # [bottom_point_x, bottom_point_y],
        # [left_point_x, left_point_y],
        # [right_point_x, right_point_y]])
        # box_x = abs(box_res[i][3][0]) # 框最大的横坐标
        # box_y = abs(box_res[i][0][1]) # 框最小的纵坐标
        # box_width = abs(box_res[i][3][0]-box_res[i][2][0])
        # box_height = abs(box_res[i][1][1]-box_res[i][0][1])
        # 浮点数要转成整数
        # 注释这个算法简直可笑
        # cx = int(box_x - box_width/2 - 10)# 向左是减
        # cy = int(box_y - box_height/2 -20) # 向上是减
        cx = int((box_res[i][3][0]+box_res[i][2][0]+box_res[i][1][0]+box_res[i][0][0])/4)
        cy = int((box_res[i][1][1]+box_res[i][0][1]+box_res[i][2][1]+box_res[i][3][1])/4)
        # print(cx,cy)
        pixel_center = hsv_frame[cy,cx]# 取hsv模型对应点的h
        hue_value=pixel_center[0]
        color = get_color(hue_value)
        cv2.putText(img=frame,text=color,org=(cx,cy),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,color=(0,255,0),thickness=2)
        cv2.circle(img=frame,center=(cx,cy),radius=5,color=(0,255,0),thickness=3)
        # 输出结果
        color_res.append(color)
    
    return color_res

# 高层调用，相当于主程序块的封装：画出图像中对应颜色的框，并返回识别结果
# findContours的返回值跟cv2版本有关，修改
def img_color_detect(frame,color_lower,color_upper):
    # for i in range(len(color_lower)):
    box_result=[] # 最终两个颜色的box数组之和
    box_num_result = 0 # 最终两个颜色box的数目之和
    box_x_result=[] # 最终两个颜色box对应的x坐标数组之和
    for i in range(len(color_lower)):
        # 初步抠图
        mask_widget,result_widget = color_cut(frame,color_lower[i],color_upper[i])
        gray = cv2.cvtColor(result_widget, cv2.COLOR_BGR2GRAY)
        ret,th = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # 图形学处理
        img = graphics_process(th)
        # cnts,_ = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)# 4.6.0版本
        _,cnts,_ = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)# 3.4.10.35版本
        # 排序
        cnts = sorted(cnts, key=cv2.contourArea,reverse=True)# 按面积由大到小
        # print(len(cnts))
        box_res,box_res_num,box_x = draw_box(frame,cnts)# 此时返回的box_res与box_x对应
        # print(box_res)
        # 结果处理
        box_result = box_result + box_res
        box_x_result = box_x_result + box_x
        box_num_result = box_num_result + box_res_num

    # print("box_x = ",box_x_result)# 打印box对应的x坐标（没有排序，与box_result对应）
    color_res = detect_box_color(frame,box_result,box_num_result)
    # color_res = detect_box_color(frame,box_res,box_res_num)
    return color_res,box_x_result

# 后续控制需要的一些数组处理，用于检索不同颜色方块对应在color_res中的index
def Get_Target_index(color_res):
    count_dict={}
    for color in color_res:
        count_dict.update({color:color_res.count(color)})

    for color in color_res :
        if count_dict[color]==1:
            # print(color)
            index = color_res.index(color)
        elif count_dict[color]==3:
            pass
        else:
            print("count error!")
            index = None
    # print(index)
    return index


# 最终该任务的顶层主函数
# 输入：两种颜色,side_cam.read()传入的图片；得到识别到物体的颜色color_res及中心x坐标，不同颜色其x坐标位置dif_color_x
def purchase_color_detect(color,frame):
    # 选择颜色
    color_lower,color_upper = choose_color(color[0],color[1])
    # print(color_lower,color_upper)
    # frame = side_camera.read()
    cv2.resize(frame,(640,480))
    height = len(frame)
    width = len(frame[0])
    # 剪裁图像下面一半
    frame = frame[int(height/2)-100:height,0:width]
    color_res,box_x_result = img_color_detect(frame,color_lower,color_upper)
    print("color_res = ",color_res) # 识别颜色结果
    print("box_x_reslut = ",box_x_result) # 对应各颜色的x坐标
    dif_color_index = Get_Target_index(color_res)
    dif_color_x = box_x_result[dif_color_index] # 不同颜色方块所处的相对坐标 
    print("dif_color_x = ", dif_color_x) # 不同颜色的物块x坐标
    # cv2.imshow('frame',frame)
    # cv2.waitKey(0)

# # ------------windows测试照片用---------------
# color_lower,color_upper = choose_color('green','yellow')
# # print(color_lower,color_upper)
# frame = cv2.imread('D:/MyWork/0BaiduAICar/data/side_pic1/16.png')
# cv2.resize(frame,(640,480))
# height = len(frame)
# width = len(frame[0])
# # 剪裁图像下面一半
# frame = frame[int(height/2)-100:height,0:width]
# color_res,box_x_result = img_color_detect(frame,color_lower,color_upper)

# print("color_res = ",color_res)
# print("box_x_reslut = ",box_x_result)
# dif_color_index = Get_Target_index(color_res)
# dif_color_x = box_x_result[dif_color_index] # 不同颜色方块所处的相对坐标 
# print("dif_color_x = ", dif_color_x)
# cv2.imshow('frame',frame)
# cv2.waitKey(0)