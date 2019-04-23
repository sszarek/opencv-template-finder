import sys
import cv2 as cv
import timeit
import file_tools as ft

args = sys.argv[1:]

if len(args) < 2:
    print('Usage: finder.py [image] [template]')
    sys.exit(1)

img_path = args[0]
template_path = args[1]

ft.assert_file_exists(img_path)
ft.assert_file_exists(template_path)

img_rgb = cv.imread(img_path)
img = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

template = cv.imread(template_path, 0)
w, h = template.shape[::-1]

methods = {
    'TM_CCOEFF': cv.TM_CCOEFF,
    'TM_CCOEFF_NORMED': cv.TM_CCOEFF_NORMED,
    'TM_CCORR': cv.TM_CCORR,
    'TM_CCORR_NORMED': cv.TM_CCORR_NORMED,
    'TM_SQDIFF': cv.TM_SQDIFF,
    'TM_SQDIFF_NORMED': cv.TM_SQDIFF_NORMED
}

for method_name, method in methods.items():
    start = timeit.default_timer()
    res = cv.matchTemplate(img, template, method)
    stop = timeit.default_timer()

    print(f'Matching using: {method_name}, took: {stop - start}ms.')

    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    img_copy = img_rgb.copy()
    cv.rectangle(img_copy, top_left, bottom_right, 255, 4)

    cv.imwrite(f'output-{method_name}.jpg', img_copy)

