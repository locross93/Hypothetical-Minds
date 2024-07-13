"""move_to()"""
import heapq


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_actions_from_path(path, start_orient):
    # used to determine move outcome based on orientation
    action_outcome_dict = {
        'N': {'FORWARD': (-1, 0), 'STEP_LEFT': (0, -1), 'STEP_RIGHT': (0, 1), 'BACKWARD': (1, 0),
            'TURN_LEFT': 'W', 'TURN_RIGHT': 'E'},
        'E': {'FORWARD': (0, 1), 'STEP_LEFT': (-1, 0), 'STEP_RIGHT': (1, 0), 'BACKWARD': (0, -1),
            'TURN_LEFT': 'N', 'TURN_RIGHT': 'S'},
        'S': {'FORWARD': (1, 0), 'STEP_LEFT': (0, 1), 'STEP_RIGHT': (0, -1), 'BACKWARD': (-1, 0),
            'TURN_LEFT': 'E', 'TURN_RIGHT': 'W'},
        'W': {'FORWARD': (0, -1), 'STEP_LEFT': (1, 0), 'STEP_RIGHT': (-1, 0), 'BACKWARD': (0, 1),
            'TURN_LEFT': 'S', 'TURN_RIGHT': 'N'},
        }    
    actions = []
    current_orient = start_orient
    for i in range(len(path) - 1):
        # y is number of rows, x is number of columns
        dy, dx = path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]
        for action, outcome in action_outcome_dict[current_orient].items():
            if outcome == (dx, dy):
                actions.append(action)
                break            
            elif isinstance(outcome, str):
                # if next state can't be reached by moving, then turn and check move again
                next_orient = outcome
                for next_action, next_outcome in action_outcome_dict[next_orient].items():
                    if next_outcome == (dx, dy):
                        actions.extend([action, next_action])
                        current_orient = next_orient
                        break
                break
    return actions


def move_to(start, goal, grid, start_orient):
    neighbors = [(0, -1), (0, 1), (-1, 0), (1, 0)]    # based on N, E, S, W orientation
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), start, start_orient))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current, current_orient = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current, current_orient = came_from[current]
                path.append(current)
            path = path[::-1]
            actions = get_actions_from_path(path, start_orient)
            return path, actions, current_orient, True

        for i, j in neighbors:
            neighbor = (current[0] + i, current[1] + j)
            # check if neighbor is in bounds and is not an obstacle (agent or wall)
            if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]) and grid[neighbor[0]][neighbor[1]] == 0:
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = (current, current_orient)
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor, current_orient))
    return [], [], '', False


"""fire_at() function"""
def get_direction(src, target):
    dx, dy = target[0] - src[0], target[1] - src[1]  # target[0] is x-axis (column), target[1] is y-axis (row)
    if abs(dx) > abs(dy):
        return 'E' if dx > 0 else 'W'
    else:
        return 'S' if dy > 0 else 'N'
    # to do is to add a case for when dx == dy, take the least actions to get to target


def turn_to_face(current_orient, target_orient):
    # outcome of turning
    turns = {
        ('N', 'E'): ['TURN_RIGHT'],
        ('N', 'W'): ['TURN_LEFT'],
        ('N', 'S'): ['TURN_RIGHT', 'TURN_RIGHT'],
        ('E', 'N'): ['TURN_LEFT'],
        ('E', 'S'): ['TURN_RIGHT'],
        ('E', 'W'): ['TURN_RIGHT', 'TURN_RIGHT'],
        ('S', 'N'): ['TURN_RIGHT', 'TURN_RIGHT'],
        ('S', 'E'): ['TURN_LEFT'],
        ('S', 'W'): ['TURN_RIGHT'],
        ('W', 'N'): ['TURN_RIGHT'],
        ('W', 'E'): ['TURN_LEFT', 'TURN_LEFT'],
        ('W', 'S'): ['TURN_LEFT'],
    }
    return turns.get((current_orient, target_orient), [])


def fire_at(src, src_orient, target):
    target_orient = get_direction(src, target)
    actions = turn_to_face(src_orient, target_orient)
    actions.append('FIRE_ZAP')
    return actions, target_orient


def interact(src, src_orient, target, grid):
    # collaborative cooking asymmetric interaction
    player_spot = 'left' if src[0] < 4 else 'right'
    if player_spot == 'left':
        obj_locs = {
            (0,1): (1,1), # tomato dispenser
            (4,2): (3,2), # pot 1
            (4,3): (3,3), # pot 2
            (3,4): (3,3), # dish dispenser
            (3,1): (3,2), # delivery location
            (2,4): (2,3), # counter
            (1,4): (1,3), # counter
            (2,1): (2,2), # counter
            (1,0): (1,1), # counter
            (0,3): (1,3), # counter
            (0,2): (1,2)  # counter
        }
    elif player_spot == 'right':
        obj_locs = {
            (5,1): (5,2), # tomato dispenser
            (4,2): (5,2), # pot 1
            (4,3): (5,3), # pot 2
            (5,4): (5,3), # dish dispenser
            (8,1): (7,1), # delivery location
            (6,1): (6,2), # counter
            (6,4): (6,3), # counter
            (7,4): (7,3), # counter
            (8,3): (7,3), # counter
            (8,2): (7,2), # counter
            (7,0): (7,1)  # counter
        }
    move_target = obj_locs.get(target, target)
    path, actions, current_orient, path_found = move_to(src, move_target, grid, src_orient)
    target_orient = get_direction(move_target, target)
    temp_actions = turn_to_face(current_orient, target_orient)
    for act in temp_actions:
        actions.append(act)
    actions.append('INTERACT')
    assert(target_orient in ['N', 'E', 'S', 'W'])
    return actions, target_orient