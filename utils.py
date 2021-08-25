import numpy as np
import time
import logging

def print_time(before):
    logging.info(time.time()-before)
    return time.time()


def linewidth_from_data_units(linewidth, axis, reference='y'):
    """
    Convert a linewidth in data units to linewidth in points.

    Parameters
    ----------
    linewidth: float
        Linewidth in data units of the respective reference-axis
    axis: matplotlib axis
        The axis which is used to extract the relevant transformation
        data (data limits and size must not change afterwards)
    reference: string
        The axis that is taken as a reference for the data width.
        Possible values: 'x' and 'y'. Defaults to 'y'.

    Returns
    -------
    linewidth: float
        Linewidth in points
    """
    fig = axis.get_figure()
    if reference == 'x':
        length = fig.bbox_inches.width * axis.get_position().width
        value_range = np.diff(axis.get_xlim())[0]
    elif reference == 'y':
        length = fig.bbox_inches.height * axis.get_position().height
        value_range = np.diff(axis.get_ylim())[0]
    # Convert length to points
    length *= 72
    # Scale linewidth to value range
    return linewidth * (length / value_range)

def add_walls_and_source_to_ax(ax, colors, S, scatter_size):

    s_mat = np.array(S)
    M, N = s_mat.shape[0], s_mat.shape[1]

    
    scatter_dict = {
        "color_item_-1_x_y":[]
    }
    for iter in range(np.sum(np.unique(s_mat) > 0)):
        scatter_dict[f"color_item_{iter+1}_x_y"] = []

    for i in range(M):
        for j in range(N):
            if S[i][j] != 0:
                scatter_dict[f"color_item_{S[i][j]}_x_y"].append([j,i])

    for key in scatter_dict:
        color_item = int(key.split('_')[2])
        mat = np.array(scatter_dict[key]).T
        j = mat[0].tolist()
        i = mat[1].tolist()
        ax.scatter(j, i, s=scatter_size*2, color=colors[color_item] if color_item != 0 else "black", marker = "x" if color_item < 0 else None)

def plot_area(S):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import colors
    ax = plt.gca()
    color = [matplotlib.colors.to_hex(c) for c in plt.cm.tab20.colors]
    s_mat = np.array(S) 
    M, N = s_mat.shape[0], s_mat.shape[1]
    
    # create discrete colormap
    # print(color[:s_mat.max()] + color[-1])
    cmap = colors.ListedColormap([color[-1]] + color[:s_mat.max()+1])
    # cmap = colors.ListedColormap(['red', 'blue', 'green', 'yellow', 'black'])
    bounds = [i for i in np.arange(-1,  s_mat.max()+2, 1)]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    ax.imshow(s_mat + 0.5, cmap=cmap, norm=norm, alpha=0.5)  # half is for the threshold

    # draw gridlines
    ax.set_facecolor('black')
    ax.set_ylim(M - 0.5, -0.5)
    ax.set_xlim(-0.5, N - 0.5)
    return ax.figure

def plot_walls(S):
    import matplotlib.pyplot as plt
    plt.clf()
    plt.cla()
    ax = plt.gca()
    colors = plt.cm.tab20.colors
    s_mat = np.array(S)
    M, N = s_mat.shape[0], s_mat.shape[1]

    

    linewidth = linewidth_from_data_units(0.45, ax)
                            
    ax.set_ylim(M - 0.5, -0.5)
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    ax.set_yticks([i + 0.5 for i in range(M - 1)], minor=True)
    ax.set_xticks([j + 0.5 for j in range(N - 1)], minor=True)
    ax.grid(b=True, which='minor', color='white')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis='both', which='both', length=0)

    add_walls_and_source_to_ax(ax, colors, S, linewidth)
    logging.info("finish plot_walls for right side")

    return ax.figure

def put_num_on_matrix(m, walls, in_num):

    matrix = np.copy(m)

    for coords in walls:
        x_min, x_max, y_min, y_max = coords[0], coords[2], coords[1], coords[3]
        for x in range(x_min, x_max + 1 if (x_min < x_max) else x_max - 1 , 1 if (x_min < x_max) else -1):
            x = int(x)
            for y in range(y_min, y_max + 1 if (y_min < y_max) else y_max - 1 , 1 if (y_min < y_max) else -1):
                y = int(y)
                if y < len(matrix) and x < len(matrix[0]) and matrix is not None:
                    matrix[y][x] = in_num

    return matrix


def put_rect_source_on_matrix(matrix, coords):


    number_of_source = 0

    x_min, x_max, y_min, y_max = coords[0], coords[2], coords[1], coords[3]
    
    for x in range(x_min, x_max + 1 if (x_min < x_max) else x_max - 1 , 1 if (x_min < x_max) else -1):
        x = int(x)
        for y in range(y_min, y_max + 1 if (y_min < y_max) else y_max - 1 , 1 if (y_min < y_max) else -1):
            y = int(y)
            matrix[y][x] = int(number_of_source/2) + 1
            number_of_source += 1
    assert number_of_source%2 == 0 and "Number of source outlet must be mutiply of 2!"
    return matrix

def put_point_source_on_matrix(matrix, coords):


    number_of_source = len(matrix)
    assert number_of_source%2 == 0 and "Number of source outlet must be mutiply of 2!"

    print(coords)
    for x,y,col in coords:
        print(x,y,col)
        matrix[int(y)][int(x)] = col

    return matrix

def create_matrix_and_area_matrix(color_dict, canvas_size, pipe_size, source, arr):

    matrix = np.zeros((np.array(canvas_size)/np.array(pipe_size)).astype('int').tolist()).astype('int')
    area_matrix = None

    color_set = list(set([item.split("_")[-1] for item in color_dict.keys()]))
    for index in color_set:
        for start_point, end_point in zip(color_dict[f"start_{index}"], color_dict[f"end_{index}"]):
            # graph.draw_rectangle(start_point, end_point, fill_color=colors[int(index)], line_color=colors[int(index)])

            start = np.array(start_point)
            end = np.array(end_point)

            start[1] = canvas_size[1]-start[1]
            end[1] = canvas_size[1]-end[1]
            start[0], start[1], end[0], end[1] = start[0]/pipe_size[0], start[1]/pipe_size[1], end[0]/pipe_size[0], end[1]/pipe_size[1]
            arr = np.append(arr, [np.append(start, end)], axis = 0) if arr is not None else np.append(start, end).reshape(-1,4)
        
        # this means wall
        if int(index) == -1:
            matrix = put_num_on_matrix(matrix, arr, in_num=-1)
        area_matrix = put_num_on_matrix(area_matrix, arr, in_num=int(index)) if area_matrix is not None else put_num_on_matrix(matrix, arr, in_num=int(index))
        arr = None

    if source:
        if len(source[0]) == 2:
            start_x = source[0][0]/pipe_size[0]
            end_x = source[1][0]/pipe_size[0]
            start_y = (canvas_size[1]-source[0][1])/pipe_size[1]
            end_y = (canvas_size[1]-source[1][1])/pipe_size[1]
            source_sized = np.array([start_x, start_y, end_x, end_y]).astype('int')
            matrix = put_rect_source_on_matrix(matrix, source_sized)
        elif len(source[0]) == 3:
            source_sized = []
            for i in range(len(source)):
                x = source[i][0]/pipe_size[0]
                y = (canvas_size[1]-source[i][1])/pipe_size[1]
                col = source[i][2]
                source_sized.append([x,y,col])

            matrix = put_point_source_on_matrix(matrix, source_sized)

    return matrix, area_matrix


if __name__ == "__main__":
    matrix = [[-1,0,1,1],
    [-1,0,2,2],
    [-1,0,0,0],
    ]
    areamatrix = [[-1,0,1,1],
    [-1,0,1,1],
    [-1,0,2,2],
    ]
    plot_walls(matrix)
    plot_area(areamatrix)
    pass