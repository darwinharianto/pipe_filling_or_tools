from ortools.sat.python import cp_model
import numpy as np
import matplotlib.pyplot as plt
import collections
import time
import traceback
import math
import sys
import utils
import logging


class VarArraySolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self, board, variables, edge, test, limit, image=None):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__variables = variables
        self.__edge = edge
        self.__solution_count = 0
        self.__board = board
        self.__test = test
        self.__before = time.time()
        self.__bg_image = image
        self.__solution_limit = limit

    def OnSolutionCallback(self):
        
        try:
            self.__solution_count += 1
            logging.info("search path time: ",  time.time()- self.__before)
            solved = []
            direction = []
            for v in self.__variables:
                # print("".join(str(self.Value(x)) for x in v))
                solved.append([self.Value(x) for x in v])
                
            for v in self.__edge:
                color_x_y_x_y = v.split(' ')[1:]
                color, y_prev, x_prev, y_cur, x_cur = color_x_y_x_y
                if (x_prev != x_cur or y_prev != y_cur) and self.Value(self.__edge[v]):
                    direction.append([color, x_prev, y_prev, x_cur, y_cur])
            self.__before = time.time()
    
            _ = plot_directed_solution_with_dir(self.__board, np.array(solved), direction)
            logging.info("test value:", self.Value(self.__test))
            logging.info("Draw time: ",  time.time()- self.__before)
            if self.__bg_image:
                ax = plt.gca()
                img = plt.imread(self.__bg_image)
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                ax.imshow(img, extent=[xlim[0], xlim[1], ylim[0], ylim[1]])
            
            ax.figure.savefig(f"/Users/darwinharianto/Documents/Git/git_training_lab/Notebook/pipe_proj/img_{self.__solution_count}.png")
            # plt.show()
            self.__before = time.time()
            logging.info("")
                    

        except Exception as e:
            logging.info(e)
            traceback.print_exc()
            raise e

        if self.__solution_count >= self.__solution_limit:
            logging.info('Stop search after %i solutions' % self.__solution_limit)
            self.StopSearch()
        

    def solution_count(self):
        return self.__solution_count

def fill_board_from_dict(S, data):
    ax = plt.gca()
    colors = plt.cm.tab20.colors
        
    s_mat = np.array(S)
    M, N = s_mat.shape[0], s_mat.shape[1]

    s_tuple = (ax.figure.get_size_inches()*ax.figure.dpi)
    s = (s_tuple[0]*s_tuple[1])/(M*N)
    
    for key in data:
        color_item = int(key.split(' ')[1])
        
        i = int(key.split(' ')[2])
        j = int(key.split(' ')[3])
        if data[key]:
            ax.scatter(j, i, s=s/4, color=colors[color_item] if color_item != 0 else "black", marker = "x")
    



def plot_directed_solution_with_dir(board, S, direction):
    
    plt.cla()
    plt.clf()
    ax = plt.gca()
    colors = plt.cm.tab20.colors
    M, N = np.array(S).shape[0], np.array(S).shape[1]
                            
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

    linewidth = utils.linewidth_from_data_units(0.45, ax)
    for item in direction:
        prev_x, prev_y = int(item[1]), int(item[2])
        cur_x, cur_y = int(item[3]),  int(item[4])
        ax.plot([prev_x, cur_x], [prev_y, cur_y], color=colors[int(item[0])], lw=linewidth)

    utils.add_walls_and_source_to_ax(ax, colors, board, linewidth**2)

    return ax.figure
  


def validate_board_and_count_colors(board):
    assert isinstance(board, list)
    assert all(isinstance(row, list) for row in board)
    assert len(set(map(len, board))) == 1
    colors = collections.Counter(square for row in board for square in row)
    del colors[0]
    neg_counter = 0
    for color in colors:
        if color < 0:
            neg_counter += 1
    for i in range(neg_counter+1):
        del colors[-i]
    assert all(count == 2 for count in colors.values())
    num_colors = len(colors)
    assert set(colors.keys()) == set(range(1, num_colors + 1))

    num_walls = len([x for sublist in board for x in sublist if x < 0])
    return num_colors, num_walls

def initiate_model_with_param(board, num_colors, num_wall=0, fill=1, area_matrix=None, min_length=0, max_length=sys.maxsize):

    # create model
    model = cp_model.CpModel()

    # create variable for model
    solution = [
        [square or model.NewIntVar(0, num_colors, "") for (j, square) in enumerate(row)]
        for (i, row) in enumerate(board)
    ]
    edge = {}
    all_turn = {}
    true = model.NewBoolVar("")
    model.AddBoolOr([true]) # without this, there will be line connecting from endpoint to endpoint after few solutions
    

    # make filled target variables
    board_size = len(board) * len(board[0])
    cus_board_size = board_size * num_colors
    at_least_filled = math.floor((board_size-num_wall)*fill)

    # make filled and turn constraint
    not_filled = 0
    turns = 0
    path_length = np.zeros(num_colors).tolist()

    # test variable
    test = model.NewIntVar(-1000, 10000, "")

    for color in range(1, num_colors + 1):
        endpoints = []
        arcs = []
        for i, row in enumerate(board):
            for j, square in enumerate(row):
                if square == color:
                    endpoints.append((i, j))
                else:
                    arcs.append(((i, j), (i, j)))
                if i < len(board) - 1:
                    if area_matrix is None:
                        if board[i+1][j] != -1: 
                            arcs.append(((i, j), (i + 1, j)))
                    else:
                        if area_matrix[i+1][j] == color:
                            arcs.append(((i, j), (i + 1, j)))
                        elif area_matrix[i+1][j] == 0:
                            arcs.append(((i, j), (i + 1, j)))
                if j < len(row) - 1:
                    if area_matrix is None:
                        if board[i][j+1] != -1: 
                            arcs.append(((i, j), (i, j + 1)))
                    else:
                        if area_matrix[i][j+1] == color or area_matrix[i][j+1] == 0:
                            arcs.append(((i, j), (i, j + 1)))
                        # elif area_matrix[i][j+1] == 0:
                        #     arcs.append(((i, j), (i, j + 1)))
        (i1, j1), (i2, j2) = endpoints
        k1 = i1 * len(row) + j1
        k2 = i2 * len(row) + j2
        arc_variables = [(k2, k1, true)] # without this, there will be line connecting from endpoint to endpoint
        
        # make length constraint
        path_length[color-1] = 0

        for (i1, j1), (i2, j2) in arcs:
            k1 = i1 * len(row) + j1
            k2 = i2 * len(row) + j2
            edge_name = f"color_i1_j1_i2_j2 {color} {i1} {j1} {i2} {j2}"
            edge[edge_name] = model.NewIntVar(0, 1, edge_name)
            if k1 == k2:
                model.Add(solution[i1][j1] != color).OnlyEnforceIf(edge[edge_name])
                arc_variables.append((k1, k1, edge[edge_name]))
                not_filled += edge[edge_name]
            else:
                model.Add(solution[i1][j1] == color).OnlyEnforceIf(edge[edge_name])
                model.Add(solution[i2][j2] == color).OnlyEnforceIf(edge[edge_name])
                forward = model.NewIntVar(0, 1, "")
                backward = model.NewIntVar(0, 1, "")
                # this clauses will force edge to be the same as forward and backward

                # Default
                model.AddBoolOr([edge[edge_name], forward.Not()])
                model.AddBoolOr([edge[edge_name], backward.Not()])
                model.AddBoolOr([edge[edge_name].Not(), forward, backward])
                model.AddBoolOr([forward.Not(), backward.Not()])

                arc_variables.append((k1, k2, forward))
                arc_variables.append((k2, k1, backward))
                path_length[color-1] += forward + backward

        for i, row in enumerate(board):
            for j, square in enumerate(row):
                bottom = f"color_i1_j1_i2_j2 {color} {i} {j} {i+1} {j}"
                top = f"color_i1_j1_i2_j2 {color} {i-1} {j} {i} {j}"
                right = f"color_i1_j1_i2_j2 {color} {i} {j} {i} {j+1}"
                left = f"color_i1_j1_i2_j2 {color} {i} {j-1} {i} {j}"

                turn_bool = model.NewIntVar(0, 1, "color_i1_j1_i2_j2")
                all_turn[f"color_i1_j1_i2_j2 {color} {i} {j} {i} {j}"] = turn_bool

                left_value = edge[left] if left in edge else true.Not()
                right_value = edge[right] if right in edge else true.Not()
                top_value = edge[top] if top in edge else true.Not()
                bottom_value = edge[bottom] if bottom in edge else true.Not()

                '''
                Minimal Form (with ~) = ~ab~cde + ~abc~de + a~b~cde + a~bc~de + ~a~b~e + ~c~d~e + cd~e + ab~e
                '''
                # minimal form
                model.AddBoolOr([left_value, right_value.Not(), bottom_value, top_value.Not(), turn_bool]) # A+ ~B + C+ ~D+ ~E
                model.AddBoolOr([left_value, right_value.Not(), bottom_value.Not(), top_value, turn_bool]) # A+ ~B + ~C+ D+ ~E
                model.AddBoolOr([left_value.Not(), right_value, bottom_value, top_value.Not(), turn_bool]) # ~A+ B + C+ ~D+ ~E
                model.AddBoolOr([left_value.Not(), right_value, bottom_value.Not(), top_value, turn_bool]) # ~A+ B + ~C+ D+ ~E
                model.AddBoolOr([left_value, right_value, turn_bool.Not()]) # A+ B + E
                model.AddBoolOr([bottom_value, top_value, turn_bool.Not()]) # C+ D+ E
                model.AddBoolOr([bottom_value.Not(), top_value.Not(), turn_bool.Not()]) # ~C+ ~D+ E
                model.AddBoolOr([left_value.Not(), right_value.Not(), turn_bool.Not()]) # ~A+ ~B+ E

                turns += turn_bool
        
        model.Add(path_length[color-1] < max_length)
        model.Add(path_length[color-1] >= min_length)
        model.AddCircuit(arc_variables)

    # model.Minimize(not_filled)
    model.Minimize(turns+not_filled)
    model.Add((cus_board_size-not_filled) >= at_least_filled)
    model.Add(test ==  cus_board_size - not_filled)

    return model, solution, edge


def solve_model(model, board, solution, edge, limit=None, image=None):
    
    # solve model
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = limit

    #### print 1 solution
    asd = time.time()
    status = solver.Solve(model)

    logging.info(f"Search sol time: {time.time()-asd}")
    # Output solution.
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        solved = []
        direction = []
        for v in solution:
            solved.append([solver.Value(x) for x in v])
        for v in edge:
            color_x_y_x_y = v.split(' ')[1:]
            color, y_prev, x_prev, y_cur, x_cur = color_x_y_x_y
            if (x_prev != x_cur or y_prev != y_cur) and solver.Value(edge[v]):
                direction.append([color, x_prev, y_prev, x_cur, y_cur])

        _ = plot_directed_solution_with_dir(board, np.array(solved), direction)
        ax = plt.gca()
        if image:
            img = plt.imread(image)
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.imshow(img, extent=[xlim[0], xlim[1], ylim[0], ylim[1]])
        ax = plt.gca()
        return ax

    else:
        logging.info("No solution found")
        return plt.gca()
    
    # #### print multiple solution
    # solution_printer = VarArraySolutionPrinter(board,solution, edge, test, limit, image=image)
    # print(f"finished preparing model {time.time()-dsa}, starting search")
    # status = solver.SearchForAllSolutions(model, solution_printer)
    # print()
    # print('Solutions found : %i' % solution_printer.solution_count())

    

def main(board, fill=1, min_length=0, max_length=sys.maxsize, limit=1, image=None, area_matrix=None, save_path=None):
    # validate board get number of wall and colors
    num_colors, num_wall = validate_board_and_count_colors(board)

    # initiate model
    model, solution, edge = initiate_model_with_param(board, num_colors, num_wall, fill=fill, area_matrix=area_matrix, min_length=min_length, max_length=max_length)

    # solve
    ax = solve_model(model, board, solution, edge, limit, image)

    plt.show()
    ax.figure.savefig(save_path) if save_path else None



if __name__ == "__main__":
    
    board_size = (10,10)
    num_of_color = 4
    
    board = np.zeros(board_size)
    for i in range(num_of_color):
        center_index = (int(board_size[0]/2)-1, int(board_size[1]/2) -1)
        
        for value in range(len(board[center_index[0]])-1):
            board[center_index[0]][value] = board[center_index[0]][value+1]
            
        for j in range(num_of_color-1):
            board[center_index[0]][center_index[1] + i] = i+1
            board[center_index[0]][center_index[1]+1 + i] = i+1
    

    board = np.array([
        [-1,1,0,0,0,0,0,0,0,0,0,0],
        [1,0,0,0,0,0,0,0,0,0,0,0],
        # [0,0,0,0,0,0,0,0,0,0,0,0],
        # [0,0,0,0,0,0,0,0,0,0,0,0],
        # [0,0,0,0,0,0,0,0,0,0,0,0],
        # [0,0,0,0,0,0,0,0,0,0,0,0],
        # [0,0,0,0,0,0,0,0,0,0,0,1],
    ])
    
    main(board.astype(int).tolist(), fill=0.8, min_length = 0, max_length = 1000, limit = 1, save_path = "./res.png")