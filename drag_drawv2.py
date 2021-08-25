from logging import error
import PySimpleGUI as sg
from numpy.lib.function_base import delete
import flow_game_solver
import traceback
from io import BytesIO
from PIL import Image
import numpy as np
import utils
import matplotlib
import matplotlib.pyplot as plt
import sys
import time
import math
import logging
import os
matplotlib.use('TkAgg')
"""
    Demo - Drag a rectangle to draw it
    This demo shows how to use a Graph Element to (optionally) display an image and then use the
    mouse to "drag a rectangle".  This is sometimes called a rubber band and is an operation you
    see in things like editors
"""

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

"""
    Apps parameter that will be used to initialize.
"""
image_file = r'~/map_sample.png'  # image is optional
sw_off = b'iVBORw0KGgoAAAANSUhEUgAAADAAAAAWCAYAAACG9x+sAAAACXBIWXMAAA7DAAAOwwHHb6hkAAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5rc2NhcGUub3Jnm+48GgAAAu5JREFUWIW9l99LU2EYx7/ve7blpm6w3cTOYnVT7hwiBJMudHVR4KQUr7qOgYhehPbjf0hIBXG7OgXdBSEyyB90UxNyIIjhPBoNodzcjWexoSic7bxdOEPKzene7XP58vDh+/A+h/c8BCUIBoNOYjL1EUJ6CHCDEeIBY42l6rlCyD5hLMmATQARQ9dnFEXJnFr670F/f7+NCcKISRBeyJJkl3w+iKIIh90OAMjmckilUlhXVagbG8jn87Vt5ogcAUYbrdax8fHxg5INDAwMiHlg5qYst3UHAnA5nWWtmqZhbmEBa/F4DTKfyjdDEHqVqamfxwd/G3gyOHjlEhALdHW5Ozs6zmWNLi5idn4ejDGOWUuyk6f0zttQaBsABOBobATgU3cgcP284QHA6/XCbDbjRyLBOeupNAuM3fN3dr6LxWJ5CgCMkOeyLLdeJPwxd/1+yJLELWU5GNC6d3j4DADI0NCQyyBka+TpU7vL5apKnMlk8Hpiom4ftqHr16huGH2yJFUdHgCcTid8LS0cslWEnVgsvZQx9pDn1ddrjACAMPaIApBFt5ub1OPxcHOdBQEkCuCyvfhI8cDB0XUWDBApd2l93oJjDAogncvluBl5uipghwJQk6kUN+M2R1cFbFAAkXVV5WZUOboqIEINXZ9ZV9XcrqZVbdMyGWxsbnLIVRFZQ9cjVFGUjFEojM4vLFRt/Dg7W69XGGDslaIoGQoAjVbr2Fo8vhJdXLyw73M0Cp6jeAYrTTbbBFD8G43FYvlb7e1zW4nEY7PF0nzV6z2X7Us0Ch43WCE7eUrvhycnNaDYAACsLi/nbre1vf+eSPjT6bRbFEXYbLaypl1Nw4fpaXxdWqpx5iKMrRom04M3odCv46P/Vsrh4WHr/sHBCBWEl5LPZ5clCR5RhMPhAABks1kkT6yUhUKhHtGzjJDR5oaG8bIr5UmCwaCTWCy9xDB6QEgLAA+AplonLbIHIAnGNhmlETMwEw6Hf59W+AeEBxzSTJhqkQAAAABJRU5ErkJggg=='
sw_on = b'iVBORw0KGgoAAAANSUhEUgAAADAAAAAWCAYAAACG9x+sAAAACXBIWXMAAA7DAAAOwwHHb6hkAAAAGXRFWHRTb2Z0d2FyZQB3d3cuaW5rc2NhcGUub3Jnm+48GgAAAr9JREFUWIXFl0tPE1EYhp9v5syAiCwgXgItiXEhlxXEhTvBtXKJW417NZoU9T9IYkE0Ju68bF1gf4CAKzcWXXDZqLEdMF7AWBBs50yPixY1Ci3SoT7LM9955/2Sc2a+V9iCSNw0OpY/iCV9GI4CEWDvVvUh8w3wgHkRk8hpZ9yLyfJmhfLnQvM9U1eT1TEM14CGXTaKY0Nv1KcnGtDWGHCgzgDwcU2YX7aZSNlMek7GDxj2AxX3YrK+ZQORuGlxLD2OcGy3jQOcbPW50p0lss+UrPNWhFvJWp6m1Cts1f/2krzbePazgSM3TTSv9HOgefcsF7AFLndnOduR+6d9j2ZdbidrFo2vjr8ekjSABYVjk1f6CVUwDzszD3CuI8fFrmxzXuknkbjZA8UG3Ky+CnSFa3NzTrb6OzK/wfnOHL2tustReghAWu6YJjfQb6jShX18erXsmS/HwqpwJlGfWcupw5Yb+INUwTwUvjaVmgdoqTeciPgNru33WyCnQvC2LXqiQchactpCpDM01TK0N+rQtDqaAgx0WBhzKDTVMuyvC0+r+MNrscKTrC75wlXKW8D7ar300/pfk0ulWosWMBuaahnmluzQtGY/2wjMWSImEZpqGSZS4TUwmbbJi0lYOe2MA5nQlEswkXZIZyo/Rt6K8GzB+aq1k7CKc/Zw5fbKo/MwNl1bsc7Ii1p0wA0vJssWgB+ouAjJipW3wdOU4tGsu+P992dcpjyVzAVqFIrDnBeTdfHVALAYjs3SjCVreLiDJh7MuNydrlkUXw1sBJv/Gmh6WzVXur4TbSg9H6Uzwmiylsm0eomtBjYNNBtE4maPY+sYcJ0qDHnKgp5ipGxvCjhYjJQf1oS5pUKknPKcrzpg2M+rkZKR8ncicdPo2n4/In3G0EYh1Nfvaje/WAU8EeYxJqEdZzx1Qb5sVvgDAJEFQLjoGwcAAAAASUVORK5CYII='

canvas_size = (700, 700)
grid_size = (10,10)
pipe_size = int(canvas_size[0]/grid_size[0]), int(canvas_size[1]/grid_size[1])
source = None
area_matrix=None
fill=0.9
min_length = 10
max_length = 10**math.floor(math.log(sys.maxsize, 10)) 
pipe_area = 1
sol_time_limit = 10
save_path = os.getcwd() + 'result.png'
COLOR_BG = sg.theme_background_color()
COLOR_BG_SWITCH = (COLOR_BG, COLOR_BG)

"""
    Apps state that will be used for display and perform action
"""
dragging = False
start_point = end_point = prior_rect = None
tmp_start = tmp_end = None
sw_state = True
start_walls = []
end_walls = []
source = []
color_dict = {
    "start_-1": [],
    "end_-1" : []
}
mode="rect"
matrix = None
arr = None
colors = [matplotlib.colors.to_hex(c) for c in plt.cm.tab20.colors]

ax = plt.gca()


settings_buttons = [
    [sg.Text('Point Mode', size=(20, 1)), sg.Button(image_data=((sw_on,sw_off)[False]), size=(20,1),button_color=COLOR_BG_SWITCH,key="-SW_1-"), sg.Text('Box Mode', size=(20, 1))],
    [sg.Text('Grid size', size=(20, 1)), sg.In(f'{grid_size}', size=(8,1), key='grid_size'),],
    [sg.Text('Fill More Than', size=(20, 1)), sg.In(f'{fill}', size=(8,1), key='fill'),],
    [sg.Text('Min Length', size=(20, 1)), sg.In(f'{min_length}', size=(8,1), key='min_length'),],
    [sg.Text('Max Length', size=(20, 1)), sg.In(f'{max_length}', size=(8,1), key='max_length'),],
    [sg.Text('Pipe Area', size=(20, 1)), sg.In(f'{pipe_area}', size=(8,1), key='pipe_area'),],
    [sg.Text('Solution Time Limit', size=(20, 1)), sg.In(f'{sol_time_limit}', size=(8,1), key='sol_time_limit'),],
    [sg.Text('Save path', size=(20, 1)), sg.In(f'{save_path}', size=(40,1), key='save_path'),],
    [sg.Button('Confirm Wall'), sg.Button('Confirm Source'), sg.Button('Confirm Area')],
    [sg.Button('Undo Wall'), sg.Button('Undo Area'), sg.Button('Clear All'),sg.Button('Generate'), sg.Button('Try SOLVER!!')],
    [sg.Button('Save'),]
]

progress_tab = [
    [sg.Text('Solving maze', size=(40, 1), key='progtext')],
    [sg.ProgressBar(3, orientation='h', size=(20, 20), key='progressbar')],
    # [sg.Cancel()]
]
layout = [
    [sg.In(key='imageFileName') ,sg.FileBrowse(file_types=(("Image Files", "*.png"),)), sg.Button('Load Image')],
    [sg.Graph(
    canvas_size=canvas_size,
    graph_bottom_left=(0, 0),
    graph_top_right=canvas_size,
    key="-GRAPH-",
    change_submits=True,  # mouse click events
    background_color='lightblue',
    drag_submits=True), sg.Graph(
    canvas_size=canvas_size,
    graph_bottom_left=(0, 0),
    graph_top_right=canvas_size,
    key="GEN",
    change_submits=True,  # mouse click events
    background_color='white',
    drag_submits=True)],
    [sg.Frame(layout=settings_buttons, title="input_frame", size=canvas_size), sg.Frame(layout=progress_tab, title="progress_frame", size=canvas_size)],

    ]


# # layout the window
# layout_progress = 


window = sg.Window("Flow Solver", layout, finalize=True)
# get the graph element for ease of use later
graph = window["-GRAPH-"]  # type: sg.Graph

im = Image.open(image_file)
with BytesIO() as output:
    # im.thumbnail(canvas_size, resample=Image.BICUBIC)
    im = im.resize(canvas_size)
    im.save(output, format="png")
    data = output.getvalue()
    
graph.draw_image(data=data, location=(0,canvas_size[1])) if image_file else None


def draw_grid(graph, canvas_size, pipe_size):
    for x in range(0, canvas_size[0], pipe_size[0]):
        graph.draw_line((x, 0), (x, canvas_size[1]))

    for y in range(canvas_size[0], 0, -pipe_size[1]):
        graph.draw_line((0, y), (canvas_size[0], y))

def draw_walls_in_graph(graph, start_walls, end_walls, index = -1):
    for start_point, end_point in zip(start_walls, end_walls):
        graph.draw_rectangle(start_point, end_point, fill_color=colors[index], line_color=colors[index])

def draw_walls_roi_in_graph(graph, color_dict):
    color_set = list(set([item.split("_")[-1] for item in color_dict.keys()]))
    
    for index in color_set:
        for start_point, end_point in zip(color_dict[f"start_{index}"], color_dict[f"end_{index}"]):
            graph.draw_rectangle(start_point, end_point, fill_color=colors[int(index)], line_color=colors[int(index)])

def refresh_image_graph(graph, image_file, canvas_size):
    im = Image.open(image_file)
    with BytesIO() as output:
        # im.thumbnail(canvas_size, resample=Image.BICUBIC)
        im = im.resize(canvas_size)
        im.save(output, format="png")
        data = output.getvalue()
    graph.erase()    
    graph.draw_image(data=data, location=(0,canvas_size[1])) if image_file else None


def refresh_window(graph, image_file=None, canvas_size=None, pipe_size=None, color_dict=None, source=None):

    logging.info("Refresh window called")
    if image_file:
        refresh_image_graph(graph, image_file, canvas_size)
        logging.info("refresh image finished")
    
    # draw_walls_in_graph(graph, color_dict["start_-1"], color_dict["end_-1"])
    draw_walls_roi_in_graph(graph, color_dict)
    logging.info("draw_walls finished")

    draw_grid(graph, canvas_size, pipe_size)
    logging.info("draw_grid finished")

    if source:
        if len(source[0]) == 2:
            graph.draw_rectangle(source[0], source[1], fill_color='blue', line_color='blue')
            logging.info("draw_rect finished")
        elif len(source[0]) == 3:
            for sou in source:
                end_point, index = (sou[0], sou[1]), sou[2]
                graph.draw_point(end_point, size=canvas_size[0]*canvas_size[1]/grid_size[0]/grid_size[1]/100, color=colors[index])
                logging.info("draw_rect finished")

    pass

    

while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED:
        break  # exit

    if sw_state:
        window.FindElement('Confirm Wall').Update(disabled=False)
        window.FindElement('Confirm Area').Update(disabled=False)
    else:
        window.FindElement('Confirm Wall').Update(disabled=True)
        window.FindElement('Confirm Area').Update(disabled=True)


    if event == "-GRAPH-":  # if there's a "Graph" event, then it's a mouse
        logging.info('Mouse event')
        x, y = values["-GRAPH-"]
        if not dragging:
            start_point = (x, y)
            dragging = True
        else:
            end_point = (x, y)
        if prior_rect:
            graph.delete_figure(prior_rect)
        if None not in (start_point, end_point) and sw_state:
            prior_rect = graph.draw_rectangle(start_point, end_point, line_color='red')
        elif None not in (start_point, end_point) and not sw_state:
            prior_rect = graph.draw_point(end_point, size=(canvas_size[0]*canvas_size[1]/grid_size[0]/grid_size[1])**0.5/2)

    elif event.endswith('+UP'):  # The drawing has ended because mouse up
        # info = window["info"]
        # info.update(value=f"grabbed rectangle from {start_point} to {end_point}")
        if sw_state:
            tmp_start = start_point
            tmp_end = end_point
            start_point, end_point = None, None  # enable grabbing a new rect
            dragging = False
        else:
            if start_point and end_point:
                pipe_area = int(values['pipe_area'])
                tmp_start = (start_point[0], start_point[1], pipe_area)
                tmp_end = (end_point[0], end_point[1], pipe_area)
                start_point, end_point = None, None  # enable grabbing a new rect
                dragging = False

    elif event == "Confirm Wall":
        if tmp_start is not None and tmp_end is not None:
            color_dict["start_-1"].append(tmp_start)
            color_dict["end_-1"].append(tmp_end)
            tmp_start, tmp_end = None, None
            refresh_window(graph, image_file, canvas_size, pipe_size, color_dict, source)

    elif event == "Save":

        ax.figure.savefig(save_path) if save_path else None


    elif event == "Generate":
        try:
            grid_size = eval(values["grid_size"])
            pipe_size = int(canvas_size[0]/grid_size[0]), int(canvas_size[1]/grid_size[1])
            assert len(pipe_size) == 2

            refresh_window(graph, image_file, canvas_size, pipe_size, color_dict, source)
            
            if len(color_dict["start_-1"]) != 0 and len(color_dict["end_-1"]) != 0 :

                matrix, area_matrix = utils.create_matrix_and_area_matrix(color_dict, canvas_size, pipe_size, source, arr)

                utils.plot_walls(matrix)
                fig = utils.plot_area(area_matrix)

                fig.canvas.draw()
                im_fig = Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())

                with BytesIO() as output:
                    # im.thumbnail(canvas_size, resample=Image.BICUBIC)
                    im_fig = im_fig.resize(canvas_size)
                    im_fig.save(output, format="png")
                    data = output.getvalue()
                
                matrix_graph = window["GEN"]   
                matrix_graph.erase()
                matrix_graph.draw_image(data=data, location=(0,canvas_size[1])) if data else None
                fig.clf()
            
        except Exception as e:
            tb = traceback.format_exc()
            sg.Print(f'An error happened.  Here is the info:', e, tb)
            sg.popup_error(f'AN EXCEPTION OCCURRED!', e, tb)

    elif event == "Load Image":
        if values['imageFileName']:
            image_file = values['imageFileName']
            # refresh_image_graph(graph, image_file, canvas_size)
            refresh_window(graph, image_file, canvas_size, pipe_size, color_dict, source)

    elif event == "Confirm Source":
        if sw_state:
            if tmp_start and tmp_end:
                source = [tmp_start,tmp_end]
                logging.info(source)
                refresh_window(graph, image_file, canvas_size, pipe_size, color_dict, source)
                tmp_start, tmp_end = None, None
        else:
            if tmp_start and tmp_end:
                source.append(tmp_end)
                logging.info(source)
                refresh_window(graph, image_file, canvas_size, pipe_size, color_dict, source)
                pass

    elif event == 'Try SOLVER!!':


        image = image_file
        limit = int(values['sol_time_limit'])
        # create the window`
        # window_progress = sg.Window('Processsing', layout_progress)
        progress_bar = window['progressbar']
        condition_list = ["Validating model....", "Building and adding constraint....", f"Solving model up to {limit} second"]


        if matrix is not None:

            board = matrix.tolist()
            fill = float(values['fill'])
            min_length = int(values['min_length'])
            max_length = int(values['max_length'])
            area_matrix = area_matrix

            # validate board get number of wall and colors
            window.Element('progtext').Update(condition_list[0])
            num_colors, num_wall = flow_game_solver.validate_board_and_count_colors(board)
            progress_bar.UpdateBar(1)

            # initiate model
            window.Element('progtext').Update(condition_list[1])
            model, solution, edge = flow_game_solver.initiate_model_with_param(board, num_colors, num_wall, fill=fill, area_matrix=area_matrix, min_length=min_length, max_length=max_length)
            progress_bar.UpdateBar(2)

            # solve
            window.Element('progtext').Update(condition_list[2])
            ax = flow_game_solver.solve_model(model, board, solution, edge, limit, image)

            progress_bar.UpdateBar(3)

            # show result on right hand side window
            fig = ax.figure
            fig.canvas.draw()
            im_fig = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())

            with BytesIO() as output:
                im_fig = im_fig.resize(canvas_size)
                im_fig.save(output, format="png")
                data = output.getvalue()
            
            # check if there is any possible solution
            if fig.get_axes():
                print("Get solution")
            matrix_graph = window["GEN"]   
            matrix_graph.erase()
            matrix_graph.draw_image(data=data, location=(0,canvas_size[1])) if data else None


    elif event == 'Confirm Area':
        # this makes area of interest
        try:
        
            if tmp_start is not None and tmp_end is not None:
                pipe_area = int(values['pipe_area'])
                print()
                if f"start_{pipe_area}" in color_dict or f"end_{pipe_area}" in color_dict:
                    color_dict[f"start_{pipe_area}"].append(tmp_start)
                    color_dict[f"end_{pipe_area}"].append(tmp_end)
                else:
                    color_dict[f"start_{pipe_area}"] = [tmp_start]
                    color_dict[f"end_{pipe_area}"] = [tmp_end]
                
                tmp_start, tmp_end = None, None
                
            refresh_window(graph, image_file, canvas_size, pipe_size, color_dict, source)

        except Exception as e:
            tb = traceback.format_exc()
            sg.Print(f'An error happened.  Here is the info:', e, tb)
            sg.popup_error(f'AN EXCEPTION OCCURRED!', e, tb)

        pass


    elif event == "Undo Wall":
        color_dict["start_-1"].pop()
        color_dict["end_-1"].pop()

        refresh_window(graph, image_file, canvas_size, pipe_size, color_dict, source)

    elif event == "Undo Area":
        pipe_area = int(values['pipe_area'])
        color_dict[f"start_{pipe_area}"].pop()
        color_dict[f"end_{pipe_area}"].pop()
        if len(color_dict[f"start_{pipe_area}"]) == 0:
            del color_dict[f"start_{pipe_area}"]
        
        if len(color_dict[f"end_{pipe_area}"]) == 0:
            del color_dict[f"end_{pipe_area}"]

        refresh_window(graph, image_file, canvas_size, pipe_size, color_dict, source)

    elif event == 'Clear All':
        for key in color_dict:
            color_dict[key] = []
        source = []
        matrix = None
        arr = None
        refresh_window(graph, image_file, canvas_size, pipe_size, color_dict, source)
    elif event.startswith("-SW"): 
        source = []
        sw_state= not sw_state
        window[event].Update(image_data=((sw_off,sw_on)[sw_state]))
    else:
        logging.info("unhandled event", event, values)