from pprint import pprint
import math
import numpy as np
import Utils

ORIGIN = [300, 0, 500, 180, 0, 180] #xyzabc, mm and degrees

def interp(xmin, xmax, ratio):
    return xmin * (1.0 - ratio) + 1.0 * ratio * xmax

def points1_gen():
    """
    generates 11 points along all 3 axes in the range [-5, 5]
    """

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
    """
    generates 11 points along all 3 axes in the range [-5, 5],
    and 10 points rotated along both horizontal ayes
    """
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

def points3_gen():
    radius = 500
    target = [300, 0, 0]
    angle_range_x = (-30, 30)
    angle_range_y = (-15, 15)
    resolution = (11, 11)

    points = []
    target_vec = np.array(target).reshape((3, 1))
    over_target_vec = np.array([0, 0, radius]).reshape((3, 1))
    # x
    for i in range(resolution[0]):
        angle_deg_x = int(interp(angle_range_x[0], angle_range_x[1], 1.0 * i / (resolution[0] - 1)))
        angle_rad_x = math.radians(angle_deg_x)

        # y
        for j in range(resolution[1]):
            angle_deg_y = int(interp(angle_range_y[0], angle_range_y[1], 1.0 * j / (resolution[1] - 1)))
            angle_rad_y = math.radians(angle_deg_y)

            tool = Utils.getTransform(0, angle_rad_y, angle_rad_x, 0, 0, 0)[:3, :3].dot(over_target_vec) + target_vec
            points.append([
                int(tool[0]),
                int(tool[1]),
                int(tool[2]),
                180 - angle_deg_x,
                -angle_deg_y,
                180])
            # print angle_deg_x, angle_deg_y
            # trf = points[-1]
            # x, y, z, a, b, c = trf
            # a, b, c = map(math.radians, (a, b, c))
            # print Utils.getTransform(c, b, a, x, y, z)
    return points, 0

def trf_points(points, trf):
    num = points[1]
    points = points[0]
    new_points = []
    for point in points:
        x, y, z, a, b, c = point
        a, b, c = map(math.radians, (a, b, c))
        point_trf = Utils.getTransform(c, b, a, x, y, z, True)
        new_trf = trf.dot(point_trf)
        c, b, a = Utils.rpy(new_trf[:3, :3])
        x = new_trf[0, 3]
        y = new_trf[1, 3]
        z = new_trf[2, 3]
        a, b, c = map(math.degrees, (a, b, c))
        x, y, z, a, b, c = map(int, (x, y, z, a, b, c))
        new_points.append([x, y, z, a, b, c])
    return new_points, num

np.set_printoptions(precision=3, suppress=True)
points1 = points1_gen()
points2 = points2_gen()
points3 = points3_gen()

if __name__ == '__main__':

    # for p in points3[0]:
    #     x, y, z, a, b, c = p
    #     a, b, c = map(math.radians, (a, b, c))
    #     print p
    #     print Utils.getTransform(c, b, a, x, y, z)

    pprint( points1)
    pprint( trf_points(points1, Utils.getTransform(np.pi, 0, 0, 0, 0, 0, True)))

    pass