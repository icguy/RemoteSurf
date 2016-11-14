from pprint import pprint

ORIGIN = [300, 0, 500, 180, 0, 180]

def points1_gen():
    points = []
    for i in range(-5, 6):
        points.append([i * 10, 0, 0, 0, 0, 0])
    for i in range(-5, 6):
        points.append([0, i * 10, 0, 0, 0, 0])
    for i in range(-5, 6):
        points.append([0, 0, i * 10, 0, 0, 0])

    points = [[p[i] + ORIGIN[i] for i in range(len(ORIGIN))] for p in points]
    return points

points1 = points1_gen()
# pprint(points1)
