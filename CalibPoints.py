from pprint import pprint
import math

ORIGIN = [300, 0, 500, 180, 0, 180]

def interp(xmin, xmax, ratio):
    return xmin * (1.0 - ratio) + 1.0 * ratio * xmax

def points1_gen():
    points = []
    for i in range(-5, 6):
        points.append([i * 10, 0, 0, 0, 0, 0])
    for i in range(-5, 6):
        points.append([0, i * 10, 0, 0, 0, 0])
    for i in range(-5, 6):
        points.append([0, 0, i * 10, 0, 0, 0])

    points = [[p[i] + ORIGIN[i] for i in range(len(ORIGIN))] for p in points]
    return points, len(points)

def points2_gen():
    points, num = points1_gen()
    num = 0

    est_height = ORIGIN[2]
    angle_range_x = (-9, 30)
    angle_range_y = (-15, 15)
    num_points_in_dir = 10

    points2 = []
    points2.append([0, 0, 0, 0, 0, 0])
    #x
    for i in range(num_points_in_dir + 1):
        angle_deg = int(interp(angle_range_x[0], angle_range_x[1], 1.0 * i / num_points_in_dir))
        angle_rad = math.radians(angle_deg)
        delta = int(math.atan(angle_rad) * est_height)
        points2.append([-delta, 0, 0, 0, angle_deg, 0])

    #y
    for i in range(num_points_in_dir + 1):
        angle_deg = int(interp(angle_range_y[0], angle_range_y[1], 1.0 * i / num_points_in_dir))
        angle_rad = math.radians(angle_deg)
        delta = int(math.atan(angle_rad) * est_height)
        points2.append([0, delta, 0, angle_deg, 0, 0])

    points2 = [[p[i] + ORIGIN[i] for i in range(len(ORIGIN))] for p in points2]
    return points + points2, num

points1 = points1_gen()
points2 = points2_gen()
pprint(points2)
