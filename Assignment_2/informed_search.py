#  Python Packages
import random
import time



#  Gap heuristics
def gap_heuristic(state):
    """Heuristic: count the number of gaps where abs difference > 1."""
    gaps = 0
    for i in range(len(state) - 1):
        if abs(state[i] - state[i + 1]) > 1:
            gaps += 1
    return gaps
#  Chebyshev heuristics
def chebyshev_heuristic(state):
    """Heuristic: max absolute distance from goal index for any pancake."""
    goal_pos = {pancake: i for i, pancake in enumerate(range(10, 0, -1))}
    max_distance = 0
    for i, pancake in enumerate(state):
        correct_index = goal_pos[pancake]
        distance = abs(i - correct_index)
        max_distance = max(max_distance, distance)
    return max_distance

# Flip top k panckes
def flip(state, k):
    """Flip top k pancakes (1-based index)."""
    return state[:k][::-1] + state[k:]

# Extract the frontier node
def get_f(node):
    return node['f']

#  A* algorithm
def search(start_state, use_heuristic):
    """A* or UCS search implementation """
    goal_state = list(range(10, 0, -1))
    start_time = time.time()
    nodes_expanded = 0
    nodes_added = 0
    max_frontier_size = 0
    if use_heuristic == 1:
        print('Using A* algorithm')
        h = gap_heuristic(start_state)
    elif use_heuristic == 2:
        print('Using UCS algorithm')
        h = 0
    else:
        print('Using Chebyshev algorithm')
        h = chebyshev_heuristic(start_state)

 # Initializing frontier state 
    frontier = [{
        'state': start_state,
        'path': [],
        'g': 0,
        'f': h  # f = g + h or f = g
    }]
    explored = set()

    while frontier:
        frontier.sort(key=get_f)
        node = frontier.pop(0)
        nodes_expanded += 1
        max_frontier_size = max(max_frontier_size, len(frontier))

        state = node['state']
        g = node['g']
        path = node['path']

        state_tuple = tuple(state)
        
        #  Checks if the node is explored
        if state_tuple in explored:
            continue
        explored.add(state_tuple)

        if state == goal_state:
            elapsed_time = time.time() - start_time
            print(f"Nodes Expanded: {nodes_expanded}")
            print(f"Nodes Added: {nodes_added}")
            print(f"Max Frontier Size: {max_frontier_size}")
            print(f"Total Time: {elapsed_time:.4f} seconds\n")
            return path, state  # Return the sequence of flips

        for k in range(2, 11):  # flip positions 2 through 10
            new_state = flip(state, k)
            new_path = path + [k]
            if tuple(new_state) in explored:
                continue

            g_new = g + 1
            # h_new = gap_heuristic(new_state) if use_heuristic else 0
            if use_heuristic == 1:
                    h_new = gap_heuristic(new_state)
            elif use_heuristic == 3:
                    h_new = chebyshev_heuristic(new_state)
            else:  # UCS
                    h_new = 0
            f_new = g_new + h_new
            # print(f"Frontier: {f_new}, Cost: {g_new}")
            frontier.append({
                'state': new_state,
                'path': new_path,
                'g': g_new,
                'f': f_new
            })
            nodes_added += 1
    return None  # No solution found

# Example usage
if __name__ == "__main__":
    random.seed(42)
    initial_state = random.sample(range(1, 11), 10)
    print("Initial state:", initial_state)
     # Choose mode
    use_heuristic = int(input("Use A* (1) or UCS (2) or Chebyshev (3)? "))

    flips, final_state = search(initial_state, use_heuristic)
    print("Flips to sort:", flips)
    print("Number of flips:", len(flips))
    print(final_state)
