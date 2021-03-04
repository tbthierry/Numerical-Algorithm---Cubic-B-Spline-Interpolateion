import numpy as np
from numpy.linalg import solve
import matplotlib.pyplot as plt
from scipy import interpolate

def compute_knots(inputData, degree):
    n = len(inputData) - 1
    num_of_knots = n + 1 + degree + degree
    head_list = [0] * degree
    tail_list = [1] * degree

    inner_list = []
    distance_sum_list = []
    distance_sum = 0
    count = 1

    while count <= n:
        DK1 = np.array(inputData[count])
        DK0 = np.array(inputData[count-1])
        distance = np.linalg.norm(DK1 - DK0)
        distance_sum += distance
        distance_sum_list.append(distance_sum)
        count += 1

    count = 0
    while count < n + 1:
        if count < 1:
            inner_list.append(0)
        elif count >= n:
            inner_list.append(1)
        else:
            inner_list.append(distance_sum_list[count-1]/distance_sum)
        count += 1

    if num_of_knots == len(head_list + inner_list + tail_list):
        return (head_list + inner_list + tail_list)
    else:
        return []

def basisFunction(knots_list, u):
    ui0 = knots_list[0]
    ui1 = knots_list[1]
    ui2 = knots_list[2]
    ui3 = knots_list[3]
    ui4 = knots_list[4]
    N_value = 0

    if u >= ui0 and u < ui1:
        N_value = ((u-ui0)**3) / ((ui1-ui0) * (ui2-ui0) * (ui3-ui0))

    if u >= ui1 and u < ui2:
        N_value = (((u-ui0)**2) * (ui2-u)) / ((ui2-ui1) * (ui3-ui0) * (ui2-ui0)) + ((ui3-u) * (u-ui0) * (u-ui1)) / ((ui2-ui1) * (ui3-ui1) * (ui3-ui0)) + ((ui4-u) * ((u-ui1)**2)) / ((ui2-ui1) * (ui4-ui1) * (ui3-ui1))

    if u >= ui2 and u < ui3:
        N_value = ((u-ui0) * ((ui3-u)**2)) / ((ui3-ui2) * (ui3-ui1) * (ui3-ui0)) + ((ui4-u) * (ui3-u) * (u-ui1)) / ((ui3-ui2) * (ui4-ui1) * (ui3-ui1)) + (((ui4-u)**2) * (u-ui2)) / ((ui3-ui2) * (ui4-ui2) * (ui4-ui1))

    if u >= ui3 and u < ui4:
        N_value = ((ui4-u)**3) / ((ui4-ui3) * (ui4-ui2) * (ui4-ui1))

    return N_value

def endpointFunction(knots_list, u):
    ui0 = knots_list[0]
    ui1 = knots_list[1]
    ui2 = knots_list[2]
    ui3 = knots_list[3]
    ui4 = knots_list[4]
    N_value = 0

    if u >= ui0 and u < ui1:
        N_value = (6*(u-ui0)) / ((ui1-ui0) * (ui2-ui0) * (ui3-ui0))

    if u >= ui1 and u < ui2:
        N_value = (2*ui2-6*u+4*ui0) / ((ui2-ui1) * (ui3-ui0) * (ui2-ui0)) + (2*ui1+2*ui0+2*ui3-6*u) / ((ui2-ui1) * (ui3-ui1) * (ui3-ui0)) + (-6*u+4*ui1+2*ui4) / ((ui2-ui1) * (ui4-ui1) * (ui3-ui1))

    if u >= ui2 and u < ui3:
        N_value = (6*u-4*ui3-2*ui0) / ((ui3-ui2) * (ui3-ui1) * (ui3-ui0)) + (6*u-2*ui1-2*ui3-2*ui4) / ((ui3-ui2) * (ui4-ui1) * (ui3-ui1)) + (6*u-4*ui4-2*ui2) / ((ui3-ui2) * (ui4-ui2) * (ui4-ui1))

    if u >= ui3 and u < ui4:
        N_value = (6*(ui4-u)) / ((ui4-ui3) * (ui4-ui2) * (ui4-ui1))

    return N_value

if __name__ == '__main__':
    # read input data
    dataPoint = []
    inputFile = open("test2.txt", "rt")
    for line in inputFile:
        dataPoint.append(list(map(float, line.strip().split(" "))))

    # default parameters
    degree = 3
    n = len(dataPoint) - 1

    # create knots and computation knots
    knots_list = compute_knots(dataPoint, degree)
    knots_list_comp = knots_list.copy()
    e = 0.000001
    for i in range(len(knots_list)-degree, len(knots_list)):
        knots_list_comp[i] = knots_list[i-1] + e

    # build N matix
    N = np.zeros([n + 3, n + 3])
    for i in range(n+1):
        for j in range(n+3):
            N[i+1][j] = basisFunction(knots_list_comp[j:j+5], knots_list_comp[i+degree])

    # Endpoints for the first row
    for i in range(n+3):
        N[0][i] = endpointFunction(knots_list_comp[i:i+5], knots_list_comp[degree])

    # Endpoints for the last -1 row
    for i in range(n+3):
        N[n + 3 - 1][i] = endpointFunction(knots_list_comp[i:i+5], knots_list_comp[-degree-1])

    # Swap rows
    N[[0, 1]] = N[[1, 0]]
    N[[n+1, n+2]] = N[[n+2, n+1]]

    # Build D matrix
    x_points = []
    y_points = []
    x_points.append(0)
    y_points.append(0)

    for xypoints in dataPoint:
        x_points.append(xypoints[0])
        y_points.append(xypoints[1])

    x_points.append(0)
    y_points.append(0)

    # swap rows
    x_points[0], x_points[1] = x_points[1], x_points[0]
    x_points[len(x_points)-2], x_points[len(x_points)-1] = x_points[len(x_points)-1], x_points[len(x_points)-2]

    y_points[0], y_points[1] = y_points[1], y_points[0]
    y_points[len(y_points) - 2], y_points[len(y_points) - 1] = y_points[len(y_points) - 1], y_points[len(y_points) - 2]

    # [N][P] = [D]
    control_points_x = [round(x, 4) for x in solve(N, x_points)]
    control_points_y = [round(x, 4) for x in solve(N, y_points)]

    # write output
    outputFile = open("test2_output.txt", "w")
    outputFile.write(str(degree) + '\n')
    outputFile.write(str(len(control_points_x)) + '\n')

    for i in knots_list:
        outputFile.write(str(i) + ' ')

    outputFile.write('\n')

    for i in range(len(control_points_x)):
        outputFile.write(str(control_points_x[i]) + ' ' + str(control_points_y[i]) + '\n')

    # plot the B-spline
    points = np.arange(0, 1, 0.01)
    inter_x = interpolate.splev(points, (knots_list, control_points_x, 3))
    inter_y = interpolate.splev(points, (knots_list, control_points_y, 3))

    plt.figure(figsize=(10, 8))
    plt.plot(np.array(dataPoint)[:, 0], np.array(dataPoint)[:, 1], 'o', label='Data Points')
    plt.plot(np.array(control_points_x), np.array(control_points_y), label='Control Polygon')
    plt.plot(inter_x, inter_y, label='Cubic B-Spline')

    plt.legend(loc=0)
    plt.tight_layout()
    # plt.savefig('test2.jpg')
    plt.show()
