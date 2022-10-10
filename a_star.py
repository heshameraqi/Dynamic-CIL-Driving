
class node:
    def __init__(self, x, y, parent=None, f=None, g=None):
        self.x = x
        self.y = y
        self.parent = parent
        self.f = f
        self.g = g

def a_star(self, map, x_start, y_start, x_goal, y_goal)
    open_list = [node(x_start, y_start, f=0, g=0)] # f shouldn't be zero but it should work anyway due to being the single item in open_list
    closed_list = []
    while not open_list:

        # Get the current node (TODO: if equal f, select least h)
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal, return the lineage of parents from s (goal) to start is the shortest path
        if current_node.x == x_goal and current_node.y == y_goal:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path

        # Loop successors (normally up to 8)
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares
            s.x = current_node.x + new_position[0]
            s.y = current_node.y + new_position[1]

            # Make sure it's within map
            if s.x > (map.shape[1] - 1) or s.x < 0 or s.y > map.shape[0] or s.y < 0:
                continue

            # Make sure not occupied cell
            if map[s.y, s.x] >= 0.5:
                continue

            g = current_node.g + ((s.x - current_node.x) ** 2) + ((s.y - current_node.y) ** 2)  # or 1 for simplicity
            h = ((s.x - x_goal) ** 2) + ((s.y - y_goal) ** 2)
            f = g + h
            s = node(s.x, s.y, parent=current_node, f=f, g=g)

            s_opened_before = get(open_list, s)
            s_closed_before = get(closed_list, s)
            if s_closed_before or (s_opened_before and s_opened_before.g < s.g):
                continue
            open_list.append(s)
