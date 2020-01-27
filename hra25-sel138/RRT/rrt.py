import sys
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import numpy as np
import copy

from operator import itemgetter
from collections import defaultdict, deque

'''
Set up matplotlib to create a plot with an empty square
'''

def setupPlot():
    fig = plt.figure(num=None, figsize=(5, 5), dpi=120,
                     facecolor='w', edgecolor='k')
    ax = fig.subplots()
    ax.set_axisbelow(True)
    ax.set_ylim(-1, 11)
    ax.set_xlim(-1, 11)
    ax.grid(which='minor', linestyle=':', alpha=0.2)
    ax.grid(which='major', linestyle=':', alpha=0.5)
    return fig, ax


'''
Make a patch for a single pology 
'''


def createPolygonPatch(polygon, color):
    verts = []
    codes = []
    for v in range(0, len(polygon)):
        xy = polygon[v]
        verts.append((xy[0], xy[1]))
        if v == 0:
            codes.append(Path.MOVETO)
        else:
            codes.append(Path.LINETO)
    verts.append(verts[0])
    codes.append(Path.CLOSEPOLY)
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor=color, lw=1)

    return patch


'''
Render the problem  
'''


def drawProblem(robotStart, robotGoal, polygons):
    _, ax = setupPlot()
    patch = createPolygonPatch(robotStart, 'green')
    ax.add_patch(patch)
    patch = createPolygonPatch(robotGoal, 'red')
    ax.add_patch(patch)
    for p in range(0, len(polygons)):
        patch = createPolygonPatch(polygons[p], 'gray')
        ax.add_patch(patch)
    plt.show()


def project(l1, l2, pt):
    line = np.array([l2 - l1])
    proj_matrix = np.matmul(line.T, line) / np.dot(l2 - l1, l2 - l1)

    proj = proj_matrix.dot(pt - l1) + l1

    if proj[0] <= l1[0]:
        proj = l1
    elif proj[0] >= l2[0]:
        proj = l2

    return proj


def get_point(tree, points, pt):
    min_dist = float("inf")
    (min_proj, r1, r2) = ((), 0, 0)

    for (v1, adj) in tree.items():
        for v2 in adj:
            l1, l2 = sorted(
                map(np.asarray, [points[v1], points[v2]]), key=itemgetter(0))

            # project point onto line
            proj = project(l1, l2, pt)

            # keep track of min distance between point and projection
            dist = np.linalg.norm(proj - pt)
            if dist < min_dist:
                min_proj = tuple(proj)
                min_dist = dist
                r1 = v1
                r2 = v2

    return (min_proj, r1, r2)


def add_point(tree, points, index, min_proj, r1, r2):
    if min_proj == points[r1]:
        tree[index].append(r1)
        tree[r1].append(index)
    elif min_proj == points[r2]:
        tree[index].append(r2)
        tree[r2].append(index)
    else:
        proj_index = len(points) + 1
        points[proj_index] = min_proj

        tree[r1].remove(r2)
        tree[r2].remove(r1)

        tree[r1].append(proj_index)
        tree[r2].append(proj_index)

        tree[proj_index].extend([r1, r2])

        tree[index].append(proj_index)
        tree[proj_index].append(index)


'''
Grow a simple RRT 
'''


def growSimpleRRT(points):
    newPoints = copy.deepcopy(points)
    adjListMap = defaultdict(list)

    adjListMap[1].append(2)
    adjListMap[2].append(1)

    # Your code goes here

    for index in points.keys():
        if index != 1 and index != 2:
            (min_proj, r1, r2) = get_point(
                adjListMap, newPoints, np.asarray(points[index]))
            add_point(adjListMap, newPoints, index, min_proj, r1, r2)

    return newPoints, dict(adjListMap)


'''
Perform basic search 
'''


def basicSearch(tree, start, goal):
    # Your code goes here. As the result, the function should
    # return a list of vertex labels, e.g.
    #
    # path = [23, 15, 9, ..., 37]
    #
    # in which 23 would be the label for the start and 37 the
    # label for the goal

    visited = set()
    queue = deque([(start,)])

    while queue:
        path = queue.popleft()
        v1 = path[-1]

        if v1 == goal:
            return list(path)
        elif v1 not in visited:
            for v2 in tree.get(v1, []):
                queue.append(path + (v2,))
            visited.add(v1)

    return []


'''
Display the RRT and Path
'''


def displayRRTandPath(points, adjListMap, path, robotStart=None, robotGoal=None, polygons=None):
    # Your code goes here
    # You could start by copying code from the function
    # drawProblem and modify it to do what you need.
    # You should draw the problem when applicable.

    if robotStart and robotGoal and polygons:
        _, ax = setupPlot()
        patch = createPolygonPatch(robotStart, 'green')
        ax.add_patch(patch)
        patch = createPolygonPatch(robotGoal, 'red')
        ax.add_patch(patch)
        for p in range(0, len(polygons)):
            patch = createPolygonPatch(polygons[p], 'gray')
            ax.add_patch(patch)

    path_edges = set([(path[i], path[i+1]) for i in range(len(path) - 1)])
    visited = set()

    for (v1, adj) in adjListMap.items():
        for v2 in adj:
            if (
                (v1, v2) not in visited
                and (v2, v1) not in visited
            ):
                visited.add((v1, v2))

                if (
                    (v1, v2) in path_edges
                    or (v2, v1) in path_edges
                ):
                    plt.plot(*zip(points[v1], points[v2]), color="orange")
                else:
                    plt.plot(*zip(points[v1], points[v2]), color="black")

    for point in points.values():
        plt.plot(*point, "ko")

    plt.show()


def shift_robot(robot, point):
    return list(map(point.__add__, robot))


def perp(a: np.array) -> np.array:
    """
    Compute orthogonal (i.e. perpendicular)
    vector of a.
    """
    b = np.empty_like(a)

    b[0] = -a[1]
    b[1] = a[0]

    return b


def intersect(x1, x2, y1, y2):
    """
    Find intersection of lines induced by vectors
    xi, yi.
    """
    dx = x2 - x1
    dy = y2 - y1

    dp = x1 - y1
    dxp = perp(dx)

    num = np.dot(dxp, dp)
    denom = np.dot(dxp, dy)

    if denom == 0:
        return np.array([])

    return (num / denom.astype(float)) * dy + y1


def is_within_bounds(x1, x2, y1, y2, int_pt):
    """
    Return true iff int_pt is within bounds of
    the line segments induced by xi, yi.
    """
    ix, iy = int_pt

    if not min(x1[0], x2[0]) <= ix <= max(x1[0], x2[0]):
        return False
    elif not min(x1[1], x2[1]) <= iy <= max(x1[1], x2[1]):
        return False
    elif not min(y1[0], y2[0]) <= ix <= max(y1[0], y2[0]):
        return False
    elif not min(y1[1], y2[1]) <= iy <= max(y1[1], y2[1]):
        return False
    else:
        return True


'''
Collision checking
'''


def isCollisionFree(robot, point, obstacles):
    # Your code goes here.
    robot = shift_robot(robot, point)

    n = len(robot)
    for i in range(n):
        (x1, x2) = (robot[i], robot[(i + 1) % n])
        x1, x2 = np.array(x1), np.array(x2)

        for obs in obstacles:
            m = len(obs)
            for j in range(m):
                (y1, y2) = (obs[j], obs[(j + 1) % m])
                y1, y2 = np.array(y1), np.array(y2)

                int_pt = intersect(x1, x2, y1, y2)
                
                if list(int_pt) and is_within_bounds(x1, x2, y1, y2, int_pt):
                    return False
    
    return True
    

def try_add_point(robot, tree, points, point, obstacles):
    (proj, r1, r2) = get_point(tree, points, point)

    if is_valid_edge(robot, proj, point, obstacles):
        index = len(points) + 1
        points[index] = tuple(point)

        add_point(tree, points, index, proj, r1, r2)
        return True

    return False


def is_valid_edge(robot, point, proj, obstacles):
    point = np.array(point)
    proj = np.array(proj)

    (closest_point, min_dist, closest_obs) = ((), float("inf"), [])

    for obs in obstacles:
        m = len(obs)
        for j in range(m):
            y1, y2 = obs[j], obs[(j + 1) % m]
            y1, y2 = np.array(y1), np.array(y2)

            int_pt = intersect(point, proj, y1, y2)

            if list(int_pt) and is_within_bounds(point, proj, y1, y2, int_pt):
                return False

            # keep track of closest obstacle
            vertex = project(point, proj, y2)
            dist = np.linalg.norm(vertex - y2)

            if dist < min_dist:
                closest_point = vertex
                min_dist = dist
                closest_obs = obs

    return isCollisionFree(robot, vertex, obstacles)


def init_points(robot, start, obstacles, goal=None):
    if goal:
        if is_valid_edge(robot, np.array(start), np.array(goal), obstacles):
            return {1: start, 2: goal}
        else:
            return {}
    else:
        while True:
            p_rand = np.random.uniform(0.0, 10.0, size=(2,))
            
            if is_valid_edge(robot, np.array(start), p_rand, obstacles):
                return {1: start, 2: tuple(p_rand)}


'''
the full RRT algorithm
'''


def rrt(robot, obstacles, startPoint, goalPoint):
    points = init_points(robot, startPoint, obstacles, goal=goalPoint)
    tree = defaultdict(list)

    if points:
        tree[1].append(2)
        tree[2].append(1)
    else:
        points = init_points(robot, startPoint, obstacles)

        tree[1].append(2)
        tree[2].append(1)
        
        while True:
            p_rand = np.random.uniform(0.0, 10.0, size=(2,))

            if isCollisionFree(robot, p_rand, obstacles):
                if try_add_point(robot, tree, points, p_rand, obstacles):
                    if try_add_point(robot, tree, points, goalPoint, obstacles):
                        break

    path = basicSearch(tree, 1, len(points))
    return points, tree, path


def main(filename, x1, y1, x2, y2, display=''):
    # read data and parse polygons
    lines = [line.rstrip('\n') for line in open(filename)]
    robot = []
    obstacles = []
    for line in range(0, len(lines)):
        xys = lines[line].split(';')
        polygon = []
        for p in range(0, len(xys)):
            xy = xys[p].split(',')
            polygon.append((float(xy[0]), float(xy[1])))
        if line == 0:
            robot = polygon
        else:
            obstacles.append(polygon)

    # print out the data
    print("robot:")
    print(str(robot))
    print("pologonal obstacles:")
    for p in range(0, len(obstacles)):
        print(str(obstacles[p]))
    print("")

    # visualize
    if display == 'display':
        robotStart = [(x + x1, y + y1) for x, y in robot]
        robotGoal = [(x + x2, y + y2) for x, y in robot]
        drawProblem(robotStart, robotGoal, obstacles)

    # example points for calling growsimplerrt
    # you should expect many mroe points, e.g., 200-500
    points = dict()
    points[1] = (5, 5)
    points[2] = (7, 8.2)
    points[3] = (6.5, 5.2)
    points[4] = (0.3, 4)
    points[5] = (6, 3.7)
    points[6] = (9.7, 6.4)
    points[7] = (4.4, 2.8)
    points[8] = (9.1, 3.1)
    points[9] = (8.1, 6.5)
    points[10] = (0.7, 5.4)
    points[11] = (5.1, 3.9)
    points[12] = (2, 6)
    points[13] = (0.5, 6.7)
    points[14] = (8.3, 2.1)
    points[15] = (7.7, 6.3)
    points[16] = (7.9, 5)
    points[17] = (4.8, 6.1)
    points[18] = (3.2, 9.3)
    points[19] = (7.3, 5.8)
    points[20] = (9, 0.6)

    # printing the points
    print("")
    print("the input points are:")
    print(str(points))
    print("")

    points, adjlistmap = growSimpleRRT(points)
    print("")
    print("the new points are:")
    print(str(points))
    print("")
    print("")
    print("the tree is:")
    print(str(adjlistmap))
    print("")

    # search for a solution
    # change 1 and 20 as you want
    path = basicSearch(adjlistmap, 1, 2)
    print("")
    print("the path is:")
    print(str(path))
    print("")

    # your visualization code
    if display == 'display':
        displayRRTandPath(points, adjlistmap, path)

    # solve a real rrt problem
    points, adjlistmap, path = rrt(robot, obstacles, (x1, y1), (x2, y2))

    # your visualization code
    if display == 'display':
        displayRRTandPath(points, adjlistmap, path,
                          robotStart, robotGoal, obstacles)


if __name__ == "__main__":
    # retrive file name for input data
    if(len(sys.argv) < 6):
        print(
            "five arguments required: python spr.py [env-file] [x1] [y1] [x2] [y2]")
        exit()

    filename = sys.argv[1]
    x1 = float(sys.argv[2])
    y1 = float(sys.argv[3])
    x2 = float(sys.argv[4])
    y2 = float(sys.argv[5])
    display = ''
    if(len(sys.argv) == 7):
        display = sys.argv[6]

    main(filename, x1, y1, x2, y2, display)
