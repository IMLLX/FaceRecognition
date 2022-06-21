import cv2
import dlib
import dlib
import cv2
import numpy as np

predictor_path = "..\\dat\\shape_predictor_68_face_landmarks.dat"
video_path = "../html/video/middleschool.mp4"
get_frame_number = 10

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

cam = cv2.VideoCapture(video_path)
cam.set(3, 1280)
cam.set(4, 720)

curr_frame = 0
fps = cam.get(5)
frame_rate = int(fps)

color_white = (255, 255, 255)
line_width = 3

lk_params_winSize = (15, 15)
lk_params_maxLevel = 2
lk_params_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

lk_params = dict(winSize=(21, 21),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def detect_track_face(_frame):
    rgb_image = cv2.cvtColor(_frame, cv2.COLOR_BGR2RGB)
    _dets = detector(rgb_image)
    _p0 = []
    for det in _dets:
        _shape = predictor(_frame, det)
        cv2.rectangle(_frame, (det.left(), det.top()), (det.right(), det.bottom()), color_white, line_width)
        for p in _shape.parts():
            cv2.circle(_frame, (p.x, p.y), 2, (0, 255, 0), -1)
            _p0.append([p.x, p.y])

    return _dets, _frame, np.array(_p0).astype('float32')


def detect_face(_frame):
    _dets = detector(_frame)
    for det in _dets:
        cv2.rectangle(_frame, (det.left(), det.top()), (det.right(), det.bottom()), color_white, line_width)
    return len(_dets) > 0, _frame


def get_face_rectangle(_dets, _shape):
    pass


def get_bounding_rect(points):
    x_points = points[:, 0].astype('int')
    y_points = points[:, 1].astype('int')
    return (x_points.min(), y_points.max()), (x_points.max(), y_points.min())


if __name__ == '__main__':
    ret_val, frame = cam.read()

    origin_frame = frame.copy()
    dets, frame, p0 = detect_track_face(frame)

    detected_frame = []
    origin_con_frame = []
    detected_frame_num = 0
    last_detected = False

    if len(dets):
        detected_frame_num += 1
        last_detected = True
        detected_frame.append(frame)
        origin_con_frame.append(origin_frame)
    while True:
        ret_val, frame = cam.read()
        if ret_val and detected_frame_num < get_frame_number:  # if next frame
            origin_frame = frame.copy()
            dets, frame, p0 = detect_track_face(frame)
            if len(dets):
                detected_frame_num += 1
                origin_con_frame.append(origin_frame)
                detected_frame.append(frame)
                if not last_detected:
                    last_detected = True
            else:
                last_detected = False
                origin_con_frame.clear()
                detected_frame.clear()
                detected_frame_num = 0
        else:
            break

    print(f'got {get_frame_number} frame..')
    print('saving...')
    for i in range(len(origin_con_frame)):
        origin_frame = origin_con_frame[i]
        detected_frame_curr = detected_frame[i]
        cv2.imwrite(f'./process_image/middleschool/recognition/frame_{i}.png', origin_frame)
        cv2.imwrite(f'./output/middleschool/recognition/frame_{i}.png', detected_frame_curr)
        good_new = []
        if i == 1:
            dets, frame, p0 = detect_track_face(origin_frame)
            good_new = p0
        else:
            old_frame = origin_con_frame[i - 1]
            old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.cvtColor(origin_frame, cv2.COLOR_BGR2GRAY)
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray,
                                                   frame_gray,
                                                   p0,
                                                   None,
                                                   maxLevel=lk_params_maxLevel,
                                                   winSize=lk_params_winSize,
                                                   criteria=lk_params_criteria
                                                   )

            good_new = np.array([p1[i] for i in range(len(p1)) if st[i] == 1]).astype('float32')
            good_new_frame = origin_frame.copy()
            p0_frame = origin_frame.copy()
            for p in good_new:
                cv2.circle(good_new_frame, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1)
            for p in p1:
                cv2.circle(p0_frame, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1)
            p0 = p1
            cv2.imwrite(f'./output/middleschool/optical/p0/frame_{i}_{len(p1)}_point.png', p0_frame)
            cv2.imwrite(f'./output/middleschool/optical/select_good/frame_{i}_{len(good_new)}_point.png', good_new_frame)
