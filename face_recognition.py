import cv2
import dlib
import dlib
import cv2
import numpy as np

predictor_path = ".\\dat\\shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
# tracker = dlib.correlation_tracker()

cam = cv2.VideoCapture(0)
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


# def detect_track_face(_frame, _tracker):
def detect_track_face(_frame):
    rgb_image = cv2.cvtColor(_frame, cv2.COLOR_BGR2RGB)
    _dets = detector(rgb_image)
    _p0 = []
    for det in _dets:
        # rectangle = dlib.rectangle(det.left(), det.top(), det.right(), det.bottom())
        # _tracker.start_track(_frame, rectangle)
        _shape = predictor(_frame, det)
        cv2.rectangle(_frame, (det.left(), det.top()), (det.right(), det.bottom()), color_white, line_width)
        for p in _shape.parts():
            cv2.circle(_frame, (p.x, p.y), 2, (0, 255, 0), -1)
            _p0.append([p.x, p.y])

    # return _dets, _frame, _tracker, np.array(_p0).astype('float32')
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

    # dets, frame, tracker, p0 = detect_track_face(frame, tracker)
    dets, frame, p0 = detect_track_face(frame)
    old_frame = frame
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('cam', frame)
    curr_frame += 1

    while True:
        ret_val, frame = cam.read()

        if ret_val:  # if next frame
            if curr_frame % frame_rate == 0:  # 1s
                dets, frame, p0 = detect_track_face(frame)
                old_frame = frame
                old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

            elif len(dets):
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # tracker.update(frame)
                # position = tracker.get_position()
                # cv2.rectangle(frame, (int(position.left()), int(position.top())),
                #               (int(position.right()), int(position.bottom())),
                #               color_white, line_width)
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray,
                                                       frame_gray,
                                                       p0,
                                                       None,
                                                       maxLevel=lk_params_maxLevel,
                                                       winSize=lk_params_winSize,
                                                       criteria=lk_params_criteria
                                                       )
                old_gray = frame_gray.copy()
                good_new = np.array([p1[i] for i in range(len(p1)) if st[i] == 1]).astype('float32')
                p0 = good_new
                for p in good_new:
                    cv2.circle(frame, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1)
                _left, _right = get_bounding_rect(good_new)
                cv2.rectangle(frame, _left,
                              _right,
                              color_white, line_width)
            curr_frame += 1
            cv2.imshow('cam', frame)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()
