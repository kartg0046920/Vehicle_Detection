import cv2
import numpy as np
import yaml
from tracker import *
from _collections import deque
from PIL import Image, ImageDraw, ImageFont
import math

# 檔案路徑
video_path = "videos/test.mp4"
save_video = "saved_videos/vehicle_detection.mp4"
yaml_file = "data/vehicle_detection.yml"
background_file = "background/background.jpg"
transform_yaml_file = "transform_data/transform_data.yml"

cap = cv2.VideoCapture(video_path)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)

object_detector = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
tracker = EuclideanDistTracker()

background = cv2.imread(background_file)
background_blur = cv2.GaussianBlur(background, (5, 5), 0)
background_gray = cv2.cvtColor(background_blur, cv2.COLOR_BGR2GRAY)

pts = [deque(maxlen=10000) for _ in range(10000)]
transformed_pts = [deque(maxlen=10000) for _ in range(10000)]

(dX, dY) = (0, 0)
direction = ""
road_direction = []
ROI_id = 0
points = []
ROI = []
bounds = []
data = []
transform_data = []
src = []
dst = []
distance = {}
tracker_1 = {}
tracker_2 = {}
speed = {}
save_data = False
drawing = False
tempFlag = False
input_direction = False

# 選擇是否儲存影片
# 選擇是否使用已儲存的偵測區域資料
# 選擇是否使用已儲存的俯視座標轉換資料
# 選擇是否使用逆向警告
config = {
    "save_video": False,
    "use_saved_yaml_data": False,
    "use_saved_transform_yaml_data": False,
    "use_reverse_warning": False
}

# 儲存影片
if config["save_video"]:
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps_cur = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(save_video, fourcc, fps_cur, size, True)
    except:
        print("[INFO] could not determine in video")

# 讀取已儲存的偵測區域資料
if config["use_saved_yaml_data"]:
    try:
        with open(yaml_file, "r") as input_data:

            if input_data != None:
                yaml_data = yaml.load(input_data)

                for i in yaml_data:
                    ROI_id = i['id'] + 1

                for input_yaml in yaml_data:
                    data.append(input_yaml)

                for i in data:
                    coordinates = np.array(i['coordinates'])
                    rect = cv2.boundingRect(coordinates)
                    ROI.append(coordinates)
                    bounds.append(rect)

                for i in data:
                    direction_restriction = i['direction']
                    road_direction.append(direction_restriction)
    except:
        pass

if config["use_saved_transform_yaml_data"]:
    try:
        with open(transform_yaml_file, "r") as input_transform_data:
            if input_transform_data != None:
                transform_yaml_data = yaml.load(input_transform_data)
                for input_transform_yaml in transform_yaml_data:
                    transform_data.append(input_transform_yaml)
        print("Using Saved Transform Data")
    except:
        pass

if transform_data == []:
    print("Using Default Transform Data")
    src = [[425, 350], [840, 350], [5, 610], [1245, 610]]
    dst = [[0, 0], [40, 0], [0, 40], [40, 40]]
    transform_width = 40
    transform_height = 40
    transform_dic = {'src': src, 'dst': dst, 'width': transform_width, 'height': transform_height}
    transform_data.append(transform_dic)
else:
    pass

# 顯示偵測區域
def draw_contours(frame,
                  coordinates,
                  border_color=(0, 0, 255)):
    cv2.drawContours(frame,
                     [coordinates],
                     contourIdx=-1,
                     color=border_color,
                     thickness=2,
                     lineType=cv2.LINE_8)


# 手動畫框
def draw_ROI(event, x, y, flags, param):
    global point, coordinates, new_coordinates, drawing, tempFlag, new_coordinates, ROI, points, ROI_id, input_direction, road_direction

    # 點擊滑鼠左鍵畫偵測框
    if event == cv2.EVENT_LBUTTONDOWN:
        tempFlag = True
        drawing = False
        point = (x, y)
        points.append((x, y))

    # 點擊滑鼠右鍵完成偵測框
    if event == cv2.EVENT_RBUTTONDOWN:
        tempFlag = True
        drawing = True

        if points != []:
            input_direction = True
        else:
            input_direction = False

    # 點擊滑鼠中鍵刪除點或線
    if event == cv2.EVENT_MBUTTONDOWN:
        tempFlag = False
        drawing = True

        if points != []:
            points.pop()

    # 雙擊滑鼠中鍵刪除偵測框
    if event == cv2.EVENT_MBUTTONDBLCLK:
        if points == [] and ROI != [] and road_direction != [] and data != []:
            ROI.pop()
            road_direction.pop()
            data.pop()

            if ROI_id > 0:
                ROI_id -= 1


cv2.namedWindow('frame')
cv2.setMouseCallback('frame', draw_ROI)


# 中文文字
def cv2ImgAddText(frame, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(frame, np.ndarray)):
        img_rbg = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    fontText = ImageFont.truetype("font/simsun.ttc", textSize, encoding="utf-8")
    draw = ImageDraw.Draw(img_rbg)
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img_rbg), cv2.COLOR_RGB2BGR)


# 輸入行駛方向
def direction_setting():
    global driving_direction, ROI_id, points, input_direction, ROI, bounds
    print("請輸入行駛方向: 0、無 1、東 2、西 3、南 4、北:")
    driving_direction = int(input())
    if driving_direction != 0 and driving_direction != 1 and driving_direction != 2 and driving_direction != 3 and driving_direction != 4:
        print("輸入錯誤，請重新輸入")
        driving_direction = int(input())
    else:
        road_direction.append(driving_direction)
        detection_data = {'id': ROI_id, 'coordinates': points, 'direction': driving_direction}
        data.append(detection_data)

        points = []
        ROI_id += 1
        ROI = []
        bounds = []

        for i in data:
            coordinates = np.array(i['coordinates'])
            rect = cv2.boundingRect(coordinates)
            new_coordinates = coordinates.reshape((-1, 1, 2))
            ROI.append(new_coordinates)
            bounds.append(rect)

        input_direction = False

        return driving_direction


while True:
    ret, frame = cap.read()
    if ret is False:
        break

    # 車速辨識範圍
    """
    cv2.line(frame, (425, 350), (840, 350), (255, 255, 255), 2)
    cv2.line(frame, (5, 610), (1245, 610), (255, 255, 255), 2)
    cv2.line(frame, (425, 350), (5, 610), (255, 255, 255), 2)
    cv2.line(frame, (840, 350), (1245, 610), (255, 255, 255), 2)
    """

    # 俯視座標轉換
    for i in transform_data:
        src = i['src']
        dst = i['dst']
        transform_width = i['width']
        transform_height = i['height']


    x_max, max_x = max(src, key=lambda item: item[0])
    x_min, min_x = min(src, key=lambda item: item[0])
    max_y, y_max = max(src, key=lambda item: item[1])
    min_y, y_min = min(src, key=lambda item: item[1])

    src_list = np.float32(src)
    dst_list = np.float32(dst)
    M = cv2.getPerspectiveTransform(src_list, dst_list)
    warped_frame = cv2.warpPerspective(frame, M, (transform_width, transform_height))
    warped_background = cv2.warpPerspective(background, M, (transform_width, transform_height))


    # 輸入行駛方向
    if input_direction == True:
        direction_setting()

    # 取得偵測區域資料
    for index, i in enumerate(data):
        coordinates = np.array(i['coordinates'])
        draw_contours(frame, coordinates)

        direction_restriction = np.array(i['direction'])
        if direction_restriction == 0:
            direction_restriction = None
        if direction_restriction == 1:
            direction_restriction = "東"
        if direction_restriction == 2:
            direction_restriction = "西"
        if direction_restriction == 3:
            direction_restriction = "南"
        if direction_restriction == 4:
            direction_restriction = "北"

        direction_moments = cv2.moments(coordinates)

        frame = cv2ImgAddText(frame, "偵測區域:" + str(i['id'] + 1),
                              int(direction_moments["m10"] / direction_moments["m00"]),
                              int(direction_moments["m01"] / direction_moments["m00"]), (255, 255, 255), 10)

        if direction_restriction == None:
            frame = cv2ImgAddText(frame, "行駛方向:不限", int(direction_moments["m10"] / direction_moments["m00"]), int(direction_moments["m01"] / direction_moments["m00"]) + 20, (255, 255, 255), 10)

        else:
            frame = cv2ImgAddText(frame, "行駛方向:" + direction_restriction, int(direction_moments["m10"] / direction_moments["m00"]), int(direction_moments["m01"] / direction_moments["m00"]) + 20, (255, 255, 255), 10)

    # 影像處理
    try:
        mask1 = np.zeros(frame.shape, np.uint8)
        mask1 = cv2.polylines(mask1, ROI, True, (0, 0, 255), thickness=2)
        roi_mask = cv2.fillPoly(mask1, ROI, (255, 255, 255))
        roi = cv2.bitwise_and(roi_mask, frame)

        roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
        roi_gray = cv2.cvtColor(roi_blur, cv2.COLOR_BGR2GRAY)

        mask2 = np.zeros(background_gray.shape, np.uint8)
        mask2 = cv2.polylines(mask2, ROI, True, (0, 0, 255), thickness=2)
        background_roi_mask = cv2.fillPoly(mask2, ROI, (255, 255, 255))
        background_roi = cv2.bitwise_and(background_roi_mask, background_gray)

        diff_roi = cv2.absdiff(roi_gray, background_roi)
        _, thresh = cv2.threshold(diff_roi, 20, 255, cv2.THRESH_BINARY)
        dilation = cv2.dilate(thresh, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 車輛偵測
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 2000 < area < 9000:
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append([x, y, w, h])

        boxes_ids = tracker.update(detections)

        for box_id in boxes_ids:
            x, y, w, h, id = box_id
            cv2.putText(frame, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            frame = cv2ImgAddText(frame, "車", int(x + (w / 2)), int(y - 30), (255, 255, 0), 20)

            center = (int(x + (w / 2)), int(y + (h / 2)))
            indexID = int(box_id[4])
            pts[indexID].append(center)

            # 俯視座標轉換
            warped_pts = np.float32(pts[indexID]).reshape(-1, 1, 2)
            transformed_pts[indexID] = cv2.perspectiveTransform(warped_pts, M)

            # 車速辨識
            if x_min < x < x_max and y_min < y < y_max and transformed_pts[indexID] is not None:

                if len(transformed_pts[indexID]) > 1:
                    distance[indexID] = math.sqrt(math.pow(transformed_pts[indexID][-1][0][0] - transformed_pts[indexID][-2][0][0], 2) + math.pow(transformed_pts[indexID][-1][0][1] - transformed_pts[indexID][-2][0][1], 2))
                    speed[indexID] = int(distance[indexID] * fps * 3.6)
                    cv2.putText(frame, str(speed[indexID]) + " km/h", (x + w, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


            # 偵測物件追蹤
            for j in np.arange(1, len(pts[indexID])):
                if pts[indexID][j - 1] is None or pts[indexID][j] is None:
                    continue
                cv2.line(frame, (pts[indexID][j - 1]), (pts[indexID][j]), (0, 255, 0), 2)

                # 方向辨識
                if len(pts[indexID]) >= 5:
                    dX = pts[indexID][-1][0] - pts[indexID][1][0]
                    dY = pts[indexID][-1][1] - pts[indexID][1][1]
                    (dirX, dirY) = ("", "")

                    if np.abs(dX) > 500:
                        dirX = "東" if np.sign(dX) == 1 else "西"

                    if np.abs(dY) > 0:
                        dirY = "南" if np.sign(dY) == 1 else "北"

                    if dirX != "" and dirY != "":
                        direction = "{}{}".format(dirX, dirY)
                    else:
                        direction = dirX if dirX != "" else dirY

                    frame = cv2ImgAddText(frame, direction, int(x + (w / 2)), y + h + 10, (0, 255, 255), 20)

                    # 逆向警告
                    if config["use_reverse_warning"]:
                        for index_direction, k in enumerate(road_direction):
                            for ind, l in enumerate(bounds):
                                if index_direction == ind and l[0] < x < int(l[0] + l[2]) and l[1] < y < int(l[1] + l[3]):
                                    if direction == "東":
                                        direction = 1
                                    if direction == "西":
                                        direction = 2
                                    if direction == "南":
                                        direction = 3
                                    if direction == "北":
                                        direction = 4
                                    if k != 0 and direction != k and direction != "":
                                        frame = cv2ImgAddText(frame, "逆向警告", x, y, (255, 0, 0), 20)
    except:
        pass

    # 滑鼠右鍵點擊
    if (tempFlag == True and drawing == False):
        if len(points) == 1:
            cv2.circle(frame, point, 2, (255, 0, 0), -2)
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], (255, 0, 0), 2)

    # 滑鼠中鍵點擊
    if (tempFlag == False and drawing == True):
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], (255, 0, 0), 2)
        if len(points) == 1:
            cv2.circle(frame, points[0], 2, (255, 0, 0), -2)

    cv2.imshow("frame", frame)

    if config["save_video"]:
        writer.write(frame)

    # 按下鍵盤 S 或 s ，直到出現 "Data Saved successfully" ，即儲存偵測區域資料
    if cv2.waitKey(1) == ord('S') or cv2.waitKey(1) == ord('s'):
        with open(yaml_file, 'w+') as output_data:
            yaml.safe_dump(data, output_data, allow_unicode=None, default_flow_style=None, sort_keys=None)
            print("Data Saved successfully")

    # 按下鍵盤 T 或 t ，直到出現 "Transform Data Saved successfully" ，即儲存俯視座標轉換資料
    if cv2.waitKey(1) == ord('T') or cv2.waitKey(1) == ord('t'):
        with open(transform_yaml_file, 'w+') as output_transform_data:
            yaml.safe_dump(transform_data, output_transform_data, allow_unicode=None, default_flow_style=None, sort_keys=None)
            print("Transform Data Saved successfully")

    # 按下鍵盤 Esc，即關閉影片
    if cv2.waitKey(1) == 27:
        break

if config["save_video"]:
    writer.release()

cap.release()
cv2.destroyAllWindows()
