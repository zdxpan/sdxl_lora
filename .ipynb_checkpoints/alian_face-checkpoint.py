# -*- coding:utf-8 _*-
"""
@file: align_face.py
@description: align and crop face, transfer landmarks accordingly
"""
import math
import cv2
from PIL import Image, ImageDraw
from matplotlib.pyplot import imshow

# import face_recognition
from collections import defaultdict
import numpy as np

def mp_align_crop_multiface(img_path, mp_face_detection):
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:
        image = cv2.imread(img_path)
        # -------------step | get keypoint -----------------------
        # process it with MediaPipe Face Detection. :input-> RGB 
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.detections:
            return None
        # Draw face detections of each face.
        height = image.shape[0]
        width = image.shape[1]
        annotated_image = image.copy()
        res = []
        for detection in results.detections:
            keypoints = dict()
            kpt_dc = {"left_eye": mp_face_detection.FaceKeyPoint.LEFT_EYE, "right_eye":mp_face_detection.FaceKeyPoint.RIGHT_EYE, 
                 "nose_tip":mp_face_detection.FaceKeyPoint.NOSE_TIP, "mouth_center":mp_face_detection.FaceKeyPoint.MOUTH_CENTER, 
                 "right_ear_tragion":mp_face_detection.FaceKeyPoint.RIGHT_EAR_TRAGION, "left_ear_tragion":mp_face_detection.FaceKeyPoint.LEFT_EAR_TRAGION}
            for k,v in kpt_dc.items():
                kpt_point = mp_face_detection.get_key_point(detection, v)
                keypoints[k] = normalized_point(width, height, kpt_point)
            keypoints['bounding_box'] = dict()
            keypoints['bounding_box']['xmin'] = int(detection.location_data.relative_bounding_box.xmin * width)
            keypoints['bounding_box']['ymin'] = int(detection.location_data.relative_bounding_box.ymin * height)
            keypoints['bounding_box']['width'] = int(detection.location_data.relative_bounding_box.width * width)
            keypoints['bounding_box']['height'] = int(detection.location_data.relative_bounding_box.height * height)
            keypoints['score'] = float(detection.score[0])
            keypoints['confidence'] = keypoints['score']
            keypoints['box'] = [keypoints['bounding_box']['xmin'], keypoints['bounding_box']['ymin'],
                                keypoints['bounding_box']['width'], keypoints['bounding_box']['height']]
            rect_start_point = (keypoints['box'][0], keypoints['box'][1])
            rect_end_point = (keypoints['box'][0]+keypoints['box'][2], keypoints['box'][1]+keypoints['box'][3])
            # ------------ step||   alian  -----------------------------
            aligned_face, eye_center, angle = align_face_mp(image_array=annotated_image, landmarks=keypoints)
            # ------------- step4 rotate keypoint 关键点 变换 -------------
            rotated_landmarks = rotate_landmarks_mp(landmarks=keypoints,
                                             eye_center=eye_center, angle=angle, row=image.shape[0])
            # draw_mp_detection(aligned_face, rotated_landmarks, (2, (12,230,111), 1))
            # ----S: --> align eye_line and key_point and box
            # stp_ = (rotated_landmarks['box'][0], rotated_landmarks['box'][1])
            # endp_ = (stp_[0]+keypoints['box'][2], stp_[1]+keypoints['box'][3])
            # print(keypoints['box'][2], keypoints['box'][3])
            # cv2.rectangle(aligned_face, stp_, endp_, (123,0,240), 0)
            # ------------ step ||    crop--------------------------------
            cropped_face, left, top = corp_face_mp(image_array=aligned_face, landmarks=rotated_landmarks)
            if top is None:
                continue
            # cv2.imwrite(person_crop_path, cropped_face)  # 保存图像文件
            res.append(cropped_face)
            
        return res
        
def mp_align_crop(img_path, mp_face_detection):
    with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:
        image = cv2.imread(img_path)
        # -------------step | get keypoint -----------------------
        # process it with MediaPipe Face Detection. :input-> RGB 
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.detections:
            return None
        # Draw face detections of each face.
        height = image.shape[0]
        width = image.shape[1]
        annotated_image = image.copy()
        for detection in results.detections:
            keypoints = dict()
            kpt_dc = {"left_eye": mp_face_detection.FaceKeyPoint.LEFT_EYE, "right_eye":mp_face_detection.FaceKeyPoint.RIGHT_EYE, 
                 "nose_tip":mp_face_detection.FaceKeyPoint.NOSE_TIP, "mouth_center":mp_face_detection.FaceKeyPoint.MOUTH_CENTER, 
                 "right_ear_tragion":mp_face_detection.FaceKeyPoint.RIGHT_EAR_TRAGION, "left_ear_tragion":mp_face_detection.FaceKeyPoint.LEFT_EAR_TRAGION}
            for k,v in kpt_dc.items():
                kpt_point = mp_face_detection.get_key_point(detection, v)
                keypoints[k] = normalized_point(width, height, kpt_point)
            keypoints['bounding_box'] = dict()
            keypoints['bounding_box']['xmin'] = int(detection.location_data.relative_bounding_box.xmin * width)
            keypoints['bounding_box']['ymin'] = int(detection.location_data.relative_bounding_box.ymin * height)
            keypoints['bounding_box']['width'] = int(detection.location_data.relative_bounding_box.width * width)
            keypoints['bounding_box']['height'] = int(detection.location_data.relative_bounding_box.height * height)
            keypoints['score'] = float(detection.score[0])
            keypoints['confidence'] = keypoints['score']
            keypoints['box'] = [keypoints['bounding_box']['xmin'], keypoints['bounding_box']['ymin'],
                                keypoints['bounding_box']['width'], keypoints['bounding_box']['height']]
            rect_start_point = (keypoints['box'][0], keypoints['box'][1])
            rect_end_point = (keypoints['box'][0]+keypoints['box'][2], keypoints['box'][1]+keypoints['box'][3])
            # ------------ step||   alian  -----------------------------
            aligned_face, eye_center, angle = align_face_mp(image_array=annotated_image, landmarks=keypoints)
            # ------------- step4 rotate keypoint 关键点 变换 -------------
            rotated_landmarks = rotate_landmarks_mp(landmarks=keypoints,
                                             eye_center=eye_center, angle=angle, row=image.shape[0])
            # draw_mp_detection(aligned_face, rotated_landmarks, (2, (12,230,111), 1))
            # ----S: --> align eye_line and key_point and box
            # stp_ = (rotated_landmarks['box'][0], rotated_landmarks['box'][1])
            # endp_ = (stp_[0]+keypoints['box'][2], stp_[1]+keypoints['box'][3])
            # print(keypoints['box'][2], keypoints['box'][3])
            # cv2.rectangle(aligned_face, stp_, endp_, (123,0,240), 0)
            # ------------ step ||    crop--------------------------------
            cropped_face, left, top = corp_face_mp(image_array=aligned_face, landmarks=rotated_landmarks)
            if top is None:
                continue
            # cv2.imwrite(person_crop_path, cropped_face)  # 保存图像文件
            return cropped_face

def Alignment_1(img,landmark):

    if landmark.shape[0]==68:
        x = landmark[36,0] - landmark[45,0]
        y = landmark[36,1] - landmark[45,1]
    elif landmark.shape[0]==5:
        x = landmark[0,0] - landmark[1,0]
        y = landmark[0,1] - landmark[1,1]
    # 眼睛连线相对于水平线的倾斜角
    if x==0:
        angle = 0
    else: 
        # 计算它的弧度制
        angle = math.atan(y/x)*180/math.pi

    center = (img.shape[1]//2, img.shape[0]//2)
    
    RotationMatrix = cv2.getRotationMatrix2D(center, angle, 1)
    # 仿射函数
    new_img = cv2.warpAffine(img,RotationMatrix,(img.shape[1],img.shape[0])) 

    RotationMatrix = np.array(RotationMatrix)
    new_landmark = []
    for i in range(landmark.shape[0]):
        pts = []    
        pts.append(RotationMatrix[0,0]*landmark[i,0]+RotationMatrix[0,1]*landmark[i,1]+RotationMatrix[0,2])
        pts.append(RotationMatrix[1,0]*landmark[i,0]+RotationMatrix[1,1]*landmark[i,1]+RotationMatrix[1,2])
        new_landmark.append(pts)

    new_landmark = np.array(new_landmark)

    return new_img, new_landmark

def align_face(image_array, landmarks):
    """ align faces according to eyes position
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return:
    rotated_img:  numpy array of aligned image
    eye_center: tuple of coordinates for eye center
    angle: degrees of rotation
    """
    # get list landmarks of left and right eye
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']
    # calculate the mean point of landmarks of left and right eye
    left_eye_center = np.mean(left_eye, axis=0).astype("int")
    right_eye_center = np.mean(right_eye, axis=0).astype("int")
    # compute the angle between the eye centroids
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    # compute angle between the line of 2 centeroids and the horizontal line
    angle = math.atan2(dy, dx) * 180. / math.pi
    # calculate the center of 2 eyes
    eye_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                  (left_eye_center[1] + right_eye_center[1]) // 2)
    eye_center = (int(eye_center[0]), int(eye_center[1]))
    # at the eye_center, rotate the image by the angle
    rotate_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
    rotated_img = cv2.warpAffine(image_array, rotate_matrix, (image_array.shape[1], image_array.shape[0]))
    return rotated_img, eye_center, angle

def normalized_point(width:int, height:int, point) -> (int, int):
    x = min(math.floor(point.x * width),  width - 1)
    y = min(math.floor(point.y * height), height - 1)
    return (x, y)

def align_face_mp(image_array, landmarks):
    """ align faces according to eyes position
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return:
    rotated_img:  numpy array of aligned image
    eye_center: tuple of coordinates for eye center
    angle: degrees of rotation
    """
    # get list landmarks of left and right eye
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']
    # calculate the mean point of landmarks of left and right eye
    # left_eye_center = np.mean(left_eye, axis=0).astype("int")
    # right_eye_center = np.mean(right_eye, axis=0).astype("int")
    # compute the angle between the eye centroids
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    # compute angle between the line of 2 centeroids and the horizontal line
    angle = 180 + math.atan2(dy, dx) * 180. / math.pi
    # calculate the center of 2 eyes
    eye_center = ((left_eye[0] + right_eye[0]) // 2,
                  (left_eye[1] + right_eye[1]) // 2)
    eye_center = (int(eye_center[0]), int(eye_center[1]))
    # at the eye_center, rotate the image by the angle
    rotate_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
    rotated_img = cv2.warpAffine(image_array, rotate_matrix, (image_array.shape[1], image_array.shape[0]))
    return rotated_img, eye_center, angle

def rotate_landmarks_mp(landmarks, eye_center, angle, row):
    """ rotate landmarks to fit the aligned face
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :param eye_center: tuple of coordinates for eye center
    :param angle: degrees of rotation
    :param row: row size of the image
    :return: rotated_landmarks with the same structure with landmarks, but different values
    """
    rotated_landmarks = defaultdict(list)
    for facial_feature in landmarks.keys():
        if facial_feature in ("bounding_box", "score", "confidence"):
            continue
        # print(facial_feature)
        if facial_feature=="box":
            point_k = (landmarks[facial_feature][0], landmarks[facial_feature][1])
            # rotated_landmark = rotate(origin=eye_center, point=point_k, angle=angle, row=row)
            # rotated_landmarks[facial_feature] = [rotated_landmark[0], rotated_landmark[1], landmarks[facial_feature][2], landmarks[facial_feature][3]]
            rotated_landmarks[facial_feature] = [point_k[0], point_k[1], landmarks[facial_feature][2], landmarks[facial_feature][3]]
            continue
        rotated_landmark = rotate(origin=eye_center, point=landmarks[facial_feature], angle=angle, row=row)
        rotated_landmarks[facial_feature].append(rotated_landmark)
    return rotated_landmarks

def corp_face_mp_box(image_array, landmarks):
    """ crop face according to eye,mouth and chin position
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return:
    cropped_img: numpy array of cropped image
    """
    stp_ = (landmarks['box'][0], landmarks['box'][1])
    endp_ = (stp_[0]+landmarks['box'][2], stp_[1]+landmarks['box'][3])
    # wd = landmarks['box'][2]
    high = landmarks['box'][3]
    pil_img = Image.fromarray(image_array)
    cropped_img = pil_img.crop((landmarks['box'][0], landmarks['box'][1], endp_[0], endp_[1]))
    cropped_img = np.array(cropped_img)
    return cropped_img

def corp_face_mp(image_array, landmarks):
    """ crop face according to eye,mouth and chin position
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return:
    cropped_img: numpy array of cropped image
    """
    stp_ = (landmarks['box'][0], landmarks['box'][1])
    endp_ = (stp_[0]+landmarks['box'][2], stp_[1]+landmarks['box'][3])
    wd = landmarks['box'][2]
    high = landmarks['box'][3]
    
    eye_landmark = np.concatenate([np.array(landmarks['left_eye']),
                                   np.array(landmarks['right_eye'])])
    eye_center = np.mean(eye_landmark, axis=0).astype("int")
    
    if 'mouth_center' not in landmarks and len(landmarks['mouth_center']) < 1:
        raise ValueError('Input landmarks or keypoints  must contain mouth_center:[(x,y)]')
    lip_center = landmarks['mouth_center'][0]
    
    mid_part = lip_center[1] - eye_center[1]
    
    top = eye_center[1] - mid_part * 2
    bottom = lip_center[1] + mid_part * 1
    bottom = max(bottom, high)

    w = h = bottom - top
    
    x_center = np.min(landmarks['mouth_center'], axis=0)[0]
    left, right = (x_center - w / 2, x_center + w / 2)
    
    if left >= right:
        print("ValueError: Coordinate 'right' is less than 'left'")
        return None,None,None

    pil_img = Image.fromarray(image_array)
    
    left, top, right, bottom = [int(i) for i in [left, top, right, bottom]]
    cropped_img = pil_img.crop((left, top, right, bottom))
    cropped_img = np.array(cropped_img)
    return cropped_img, left, top

def draw_mp_detection(
    image: np.ndarray,
    landmarks: defaultdict(list),
    point_color=(1, (12,230,0), 1),
    box_color=((12,230,0), 0)):
    """Draws the detction bounding box and keypoints on the image.
    Args:
      image: A three channel BGR image represented as numpy ndarray.
      landmarks: A landmarks msg to be annotated on the image.
    """
    for facial_feature in landmarks.keys():
        if facial_feature in ("bounding_box", "score", "confidence", 'box'):
            continue
        point_k = landmarks[facial_feature][0]
        cv2.circle(image, point_k, point_color[0],
               point_color[1], point_color[2])

    facial_feature = 'box'
    x,y = landmarks['box'][0], landmarks['box'][1]
    st,end = x+landmarks[facial_feature][2],y+landmarks[facial_feature][3]
    cv2.rectangle(image, rect_start_point, rect_end_point,
                box_color[0], box_color[1])

def align_face_mtcnn(image_array, landmarks):
    """ align faces according to eyes position
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return:
    rotated_img:  numpy array of aligned image
    eye_center: tuple of coordinates for eye center
    angle: degrees of rotation
    """
    # get list landmarks of left and right eye
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']
    # calculate the mean point of landmarks of left and right eye
    #left_eye_center = np.mean(left_eye, axis=0).astype("int")
    #right_eye_center = np.mean(right_eye, axis=0).astype("int")
    # compute the angle between the eye centroids
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    # compute angle between the line of 2 centeroids and the horizontal line
    angle = math.atan2(dy, dx) * 180. / math.pi
    # calculate the center of 2 eyes
    eye_center = ((left_eye[0] + right_eye[0]) // 2,
                  (left_eye[1] + right_eye[1]) // 2)
    eye_center = (int(eye_center[0]), int(eye_center[1]))
    # at the eye_center, rotate the image by the angle
    rotate_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
    rotated_img = cv2.warpAffine(image_array, rotate_matrix, (image_array.shape[1], image_array.shape[0]))
    return rotated_img, eye_center, angle

def rotate(origin, point, angle, row):
    """ rotate coordinates in image coordinate system
    :param origin: tuple of coordinates,the rotation center
    :param point: tuple of coordinates, points to rotate
    :param angle: degrees of rotation
    :param row: row size of the image
    :return: rotated coordinates of point
    """
    x1, y1 = point
    x2, y2 = origin
    y1 = row - y1
    y2 = row - y2
    angle = math.radians(angle)
    x = x2 + math.cos(angle) * (x1 - x2) - math.sin(angle) * (y1 - y2)
    y = y2 + math.sin(angle) * (x1 - x2) + math.cos(angle) * (y1 - y2)
    y = row - y
    return int(x), int(y)


def rotate_landmarks(landmarks, eye_center, angle, row):
    """ rotate landmarks to fit the aligned face
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :param eye_center: tuple of coordinates for eye center
    :param angle: degrees of rotation
    :param row: row size of the image
    :return: rotated_landmarks with the same structure with landmarks, but different values
    """
    rotated_landmarks = defaultdict(list)
    for facial_feature in landmarks.keys():
        for landmark in landmarks[facial_feature]:
            rotated_landmark = rotate(origin=eye_center, point=landmarks, angle=angle, row=row)
            rotated_landmarks[facial_feature].append(rotated_landmark)
    return rotated_landmarks

def rotate_landmarks_mtcnn(landmarks, eye_center, angle, row):
    """ rotate landmarks to fit the aligned face
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :param eye_center: tuple of coordinates for eye center
    :param angle: degrees of rotation
    :param row: row size of the image
    :return: rotated_landmarks with the same structure with landmarks, but different values
    """
    rotated_landmarks = defaultdict(list)
    for facial_feature in landmarks.keys():
        rotated_landmark = rotate(origin=eye_center, point=landmarks[facial_feature], angle=angle, row=row)
        rotated_landmarks[facial_feature].append(rotated_landmark)
    return rotated_landmarks


def corp_face(image_array, landmarks):
    """ crop face according to eye,mouth and chin position
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return:
    cropped_img: numpy array of cropped image
    """

    eye_landmark = np.concatenate([np.array(landmarks['left_eye']),
                                   np.array(landmarks['right_eye'])])
    eye_center = np.mean(eye_landmark, axis=0).astype("int")
    if 'top_lip' in landmarks:
        lip_landmark = np.concatenate([np.array(landmarks['top_lip']),
                                   np.array(landmarks['bottom_lip'])])
    else:
        lip_landmark = np.concatenate([np.array(landmarks['mouth_left']),
                                   np.array(landmarks['mouth_right'])])
    lip_center = np.mean(lip_landmark, axis=0).astype("int")
    mid_part = lip_center[1] - eye_center[1]
    top = eye_center[1] - mid_part * 30 / 35
    bottom = lip_center[1] + mid_part

    w = h = bottom - top
    x_min = np.min(landmarks['chin'], axis=0)[0]
    x_max = np.max(landmarks['chin'], axis=0)[0]
    x_center = (x_max - x_min) / 2 + x_min
    left, right = (x_center - w / 2, x_center + w / 2)

    pil_img = Image.fromarray(image_array)
    left, top, right, bottom = [int(i) for i in [left, top, right, bottom]]
    cropped_img = pil_img.crop((left, top, right, bottom))
    cropped_img = np.array(cropped_img)
    return cropped_img, left, top

def corp_face_mtcnn(image_array, landmarks):
    """ crop face according to eye,mouth and chin position
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return:
    cropped_img: numpy array of cropped image
    """

    eye_landmark = np.concatenate([np.array(landmarks['left_eye']),
                                   np.array(landmarks['right_eye'])])
    eye_center = np.mean(eye_landmark, axis=0).astype("int")
    if 'top_lip' in landmarks:
        lip_landmark = np.concatenate([np.array(landmarks['top_lip']),
                                   np.array(landmarks['bottom_lip'])])
    else:
        lip_landmark = np.concatenate([np.array(landmarks['mouth_left']),
                                   np.array(landmarks['mouth_right'])])
    lip_center = np.mean(lip_landmark, axis=0).astype("int")
    print(lip_center)
    mid_part = lip_center[1] - eye_center[1]
    top = eye_center[1] - mid_part * 30 / 35
    bottom = lip_center[1] + mid_part

    w = h = bottom - top
    x_min = np.min(landmarks['mouth_left'], axis=0)[0]
    x_max = np.max(landmarks['mouth_right'], axis=0)[0]
    x_center = (x_max - x_min) / 2 + x_min
    left, right = (x_center - w / 2, x_center + w / 2)

    pil_img = Image.fromarray(image_array)
    left, top, right, bottom = [int(i) for i in [left, top, right, bottom]]
    cropped_img = pil_img.crop((left, top, right, bottom))
    cropped_img = np.array(cropped_img)
    return cropped_img, left, top



def transfer_landmark(landmarks, left, top):
    """transfer landmarks to fit the cropped face
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :param left: left coordinates of cropping
    :param top: top coordinates of cropping
    :return: transferred_landmarks with the same structure with landmarks, but different values
    """
    transferred_landmarks = defaultdict(list)
    for facial_feature in landmarks.keys():
        for landmark in landmarks[facial_feature]:
            transferred_landmark = (landmark[0] - left, landmark[1] - top)
            transferred_landmarks[facial_feature].append(transferred_landmark)
    return transferred_landmarks


def face_process(image, landmark_model_type='large'):
    """ for a given image, do face alignment and crop face
    :param image: numpy array of a single image
    :param landmark_model_type: 'large' returns 68 landmarks; 'small' return 5 landmarks
    :return:
    cropped_face: image array with face aligned and cropped
    transferred_landmarks: landmarks that fit cropped_face
    """
    # detect landmarks
    face_landmarks_dict = detect_landmark(image_array=image, model_type=landmark_model_type)
    # rotate image array to align face
    aligned_face, eye_center, angle = align_face(image_array=image, landmarks=face_landmarks_dict)
    # rotate landmarks coordinates to fit the aligned face
    rotated_landmarks = rotate_landmarks(landmarks=face_landmarks_dict,
                                         eye_center=eye_center, angle=angle, row=image.shape[0])
    # crop face according to landmarks
    cropped_face = corp_face(image_array=aligned_face, landmarks=rotated_landmarks)
    # transfer landmarks to fit the cropped face
    transferred_landmarks = transfer_landmark(landmarks=rotated_landmarks, left=left, top=top)
    return cropped_face, transferred_landmarks


def visualize_landmark(image_array, landmarks):
    """ plot landmarks on image
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return: plots of images with landmarks on
    """
    origin_img = Image.fromarray(image_array)
    draw = ImageDraw.Draw(origin_img)
    for facial_feature in landmarks.keys():
        draw.point(landmarks[facial_feature])
    imshow(origin_img)

def visualize_landmark2(image_array, landmarks):
    """ plot landmarks on image
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return: plots of images with landmarks on
    """
    origin_img = Image.fromarray(image_array)
    draw = ImageDraw.Draw(origin_img)
    for facial_feature in landmarks.keys():
        if facial_feature in ("bounding_box", "score", "confidence"):
            continue
        draw.point(landmarks[facial_feature])
    imshow(origin_img)

if __name__ == '__main__':
    # load image
    img_name = 'Messi_align.png'
    # img_name = './img/Britney_Spears_0004.jpg'

    img = cv2.imread(img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # process the face image
    face, landmarks = face_preprocess(image=image_array,
                                      landmark_model_type='large',
                                      crop_size=140)

    visualize_landmark(image_array=face, landmarks=landmarks)
    plt.show()