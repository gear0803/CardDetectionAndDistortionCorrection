#校园卡定位以及动态矫正

import cv2
import numpy as np
from numpy.core.fromnumeric import shape, sort


def my_cross_point(line1, line2, size):
    '''
    求两条直线的交点和是否平行
    line1,line2:每条line用两个点的xy坐标表示
    size：图像的二维尺寸
    out_x,out_y：交点的坐标。如果两直线交点超出图像范围则返回-1
    is_parallel:True-两直线近似平行，交点在图像外；False：两直线在图像内有交点
    '''

    [[x1, y1, x2, y2]] = line1
    [[x3, y3, x4, y4]] = line2
    x_max, y_max = size
    '''
    这里加一个斜率为0或是斜率不存在的判断
    '''
    if(abs(x1 - x2) < 1 or abs(x3 - x4) < 1):  # 两条线中至少一条是垂直线
        if(abs(x1 - x2) < 1 and abs(x3-x4) > 1):  # 只有line1是垂直线
            out_x = x1
            out_y = (y3-y4)/(x3-x4)*(x1-x3)+y3  # 也就是k3*(x-x3)+y3
            is_parallel = False
            # return out_x, out_y, is_parallel
        elif(abs(x1 - x2) > 1 and abs(x3-x4) < 1):  # 只有line2是垂直线
            out_x = x3
            out_y = (y1-y2)/(x1-x2)*(x3-x1)+y1  # 也就是k1*(x-x1)+y1
            is_parallel = False

        else:  # 两条都是垂直线
            out_x = -1
            out_y = -1
            is_parallel = True
    else:  # 两条都不是垂直线 斜率都存在 代直线公式解坐标
        k1 = (y1-y2)/(x1-x2)
        k3 = (y3-y4)/(x3-x4)
        is_parallel = False  # 默认不平行

        # k1 = k3时 直接返回平行，不需要再计算
        if(abs(k1-k3) < 1):
            out_x = -1
            out_y = -1
            is_parallel = True

            return out_x, out_y, is_parallel

        # k1!=k3
        out_x = (k1*x1-k3*x3-y1+y3)/(k1-k3)
        out_y = k1*(out_x-x1)+y1

        # 如果输出的xy超出了图像本身的范围 说明两者是近乎平行的关系
        if(out_x <= 0 or out_y <= 0 or out_x > x_max or out_y > y_max):
            out_x = -1
            out_y = -1
            is_parallel = True
        # return out_x, out_y, is_parallel

    return out_x, out_y, is_parallel


cap = cv2.VideoCapture("1.mp4")
while(cap.isOpened()):
    print("open")
    ret, videoframe = cap.read()
    cv2.imshow('videoframe', videoframe)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    # 对videoframe做处理
    img = videoframe
    # ---- 图像处理 ----

    # 使用颜色HSV进行提取
    mat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 由于校园卡会存在不同程度的褪色 因此只划定了色调范围，亮度和饱和度都不指定
    low_blue = np.array([78, 0, 0])
    upper_blue = np.array([124, 255, 255])
    mask = cv2.inRange(mat, low_blue, upper_blue)
    ROI = mask
    SE = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    ROI_open = cv2.morphologyEx(ROI, cv2.MORPH_CLOSE, (5, 5))
    # 用小SE腐蚀，再和原图做差，得到外轮廓
    SE_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    ROI_erode = cv2.erode(ROI_open, SE_small)
    ROI_edge = ROI_open-ROI_erode
    # 开闭操作 去除少量孤立点
    ROI_edge = cv2.morphologyEx(ROI_edge, cv2.MORPH_CLOSE, SE)
    ROI_edge = cv2.morphologyEx(
        ROI_edge, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)))
    # 图像中可能有其他杂物 因此需要从所有轮廓中找到最大的那个作为校园卡的轮廓
    contours_none = cv2.findContours(
        ROI_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    biggest = sorted(contours_none, key=lambda x: cv2.contourArea(
        x), reverse=True)  # 根据contour的面积大小排序
    drawing_none = cv2.drawContours(img, biggest, 0, (0, 255, 0), 3)
    #cv2.imshow("处理后", img)

    # ---- 找到目标顶点 ----

    # 用多边形拟合校园卡的外轮廓 其中epsilon是参数，可以调整拟合的精度
    epsilon = 0.01
    approx = cv2.approxPolyDP(
        biggest[0], epsilon * cv2.arcLength(biggest[0], True), True)
    boxPoints = approx.reshape(-1, 2)
    # 画出拟合多边形的顶点和边
    num = len(approx)  # 顶点数目 = 边数
    for i in range(0, num):
        cv2.line(img, boxPoints[i], boxPoints[(i+1) % num], (255, 100, 100), 4)
    for i in range(0, num):
        cv2.circle(img, boxPoints[i], 10, (240, 240, 200), -1)
    # cv2.imshow("多边形",img)
    # 拟合得到的多边形不一定是四边形，可能有很多条边。我们选择长度最长的四条边作为校园卡的四条边
    mysidelen = np.ones((num, 1))
    for i in range(0, num):
        mysidelen[i] = np.linalg.norm(boxPoints[(i+1) % num]-boxPoints[i])

    four_side_len = sorted(mysidelen, reverse=True)[0:4]
    # 得到最长的四条边的顶点编号，[编号-编号+1]即是该边的两个端点 %num是为了处理边界情况
    pointID = sorted(range(num), key=lambda k: mysidelen[k], reverse=True)[0:4]
    four_side = np.ones((4, 1, 4))  # 4行4列 每一个元素是构成一条直线的两个点的坐标 后面算交点要用
    for i in range(0, 4):
        cv2.line(img, boxPoints[pointID[i]],
                 boxPoints[(pointID[i]+1) % num], (0, 0, 255), 4)
        four_side[i] = np.array([boxPoints[pointID[i]][0], boxPoints[pointID[i]][1], boxPoints[(
            pointID[i]+1) % num][0], boxPoints[(pointID[i]+1) % num][1]])

    # 校园卡的长宽比大概是1.57左右，因此大部分的情况下，图像中最长的边一般就是校园卡的长
    # four_side是按长度降序排序的。因此four_side[0]指示的一定是最长边
    # 其他的边无法用长度确定，因为在透视扭曲得很严重的情况下，长边可能会比宽边短
    # 判断最长边和其他三条边判断是否平行。若不平行则是宽边，可以找到两个交点
    # 若平行则是长边，则接下来用平行的这条边和两条宽边相交，找到剩下的两个交点

    cross_points = np.zeros((4, 2))
    j = 0
    for i in range(1, 4):
        temp_x, temp_y, is_parallel = my_cross_point(
            four_side[0], four_side[i], (1080, 1440))  # (np.shape(img)[1], np.shape(img)[0]))
        if(is_parallel):  # 找到平行边
            parallel_index = i
            continue
        cross_points[j, :] = temp_x, temp_y
        j += 1
    # parallel_index：和长边平行的那条边的编号

    # 找平行边和其他两条边的交点
    j = 2
    for i in range(1, 4):
        if(i == parallel_index):
            continue
        temp_x, temp_y, is_parallel = my_cross_point(
            four_side[parallel_index], four_side[i], (1080, 1440))  # (np.shape(img)[1], np.shape(img)[0]))
        cross_points[j, :] = temp_x, temp_y
        j += 1

    # 在图上标出四个点
    for i in range(0, 4):
        cv2.circle(img, np.array(
            [cross_points[i][0], cross_points[i][1]], dtype=np.int32), 5, (0, 0, 0), -1)
    #cv2.imshow("box2", img)

        # 需要找到四个交点在图像中的相对方位，按照“左上 左下 右上 右下”的顺序排列
    # 方法：先按列排序，可以把顶点分成左右两组，再在组内按行排序，分出上下
    sorted_cross_points = sorted(cross_points, key=lambda x: x[0])
    sorted_cross_points = sorted(sorted_cross_points[0:2], key=lambda x: x[1]) + sorted(
        sorted_cross_points[2:4], key=lambda x: x[1])

    # 由于目标映射的点需要按照顺时针排列，即“左上 右上 右下 左下”
    # 然而由于卡可能是竖着放的，因此图像中的左上角不一定是卡的左上角，所以需要分类处理
    # horizontal：卡是横着放的 vertical：卡是竖着放的
    indices_horizontal = [0, 2, 3, 1]  # 左上 右上 右下 左下（顺时针）的顺序
    indices_vertical = [1, 0, 2, 3]

    # 判断标准：左上离左下更近为横着放 左上离右上更近-竖着放
    h_or_v = np.linalg.norm(sorted_cross_points[0]-sorted_cross_points[1]) < np.linalg.norm(
        sorted_cross_points[0]-sorted_cross_points[2])
    if(h_or_v):  # 横着放
        sorted_cross_points = np.array(sorted_cross_points)[
            indices_horizontal[:]]
        for i in range(0, 4): #画线
            cv2.line(img, np.array(cross_points[indices_horizontal[i]], dtype=np.int32), np.array(
                cross_points[indices_horizontal[(i+1) % 4]], dtype=np.int32), (0, 0, 255), 4)

    else:  # 竖着放
        sorted_cross_points = np.array(sorted_cross_points)[
            indices_vertical[:]]
        for i in range(0, 4): #画线
            cv2.line(img, np.array(cross_points[indices_horizontal[i]], dtype=np.int32), np.array(
                cross_points[indices_horizontal[(i+1) % 4]], dtype=np.int32), (0, 0, 255), 4)

    cv2.imshow("fixed",img)
    # ---- 透视变换 ----

    # 校园卡的尺寸大概是850mm*540mm
    # 求出透视矩阵并做透视变换
    dstPoints = np.array(
        [[0, 0], [850, 0], [850, 540], [0, 540]], dtype=np.int32)
    M = cv2.findHomography(sorted_cross_points, dstPoints)
    Ma = np.array(M[0], dtype=np.float64)

    # ---- 结果展示 ----
    dst_img = img
    img_f64 = np.float64(dst_img)
    output = cv2.warpPerspective(dst_img, M[0], (850, 540))

    cv2.imshow("outpu", output)
   

cap.release()
cv2.destroyAllWindows()

print("fin")
