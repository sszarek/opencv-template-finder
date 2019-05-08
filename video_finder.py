import cv2
import sys

from file_tools import assert_file_exists

args = sys.argv[1:]

if len(args) < 1:
    print('Usage: video_finder.py [template]')
    sys.exit(1)

template_path = args[0]
assert_file_exists(template_path)

template = cv2.imread(template_path, 0)
w, h = template.shape[::-1]
template = cv2.resize(template, (int(w * 0.5), int(h * 0.5)))
w, h = template.shape[::-1]
print(f'Pattern - Width: {w}, height: {h}')

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        method = cv2.TM_CCOEFF
        res = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        frame_copy = frame.copy()
        cv2.rectangle(frame_copy, top_left, bottom_right, 255, 4)

        cv2.imshow('original', frame)
        cv2.imshow('grayscale', frame_gray)
        cv2.imshow('result', frame_copy)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
