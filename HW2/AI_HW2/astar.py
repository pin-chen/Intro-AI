import csv
#
import queue
#
edgeFile = 'edges.csv'
heuristicFile = 'heuristic.csv'


def astar(start, end):
    # Begin your code (Part 4)
    #raise NotImplementedError("To be implemented")
    """
        1. Load all data in edges.csv with { start node id : { end node id : (distance, speed limit) } }
        2. init node_visted as -10
    """
    graph = dict()
    node_visted = dict()
    with open(edgeFile) as f:
        for line in f:
            data = line.split(',')
            if data[0] == 'start':
                continue
            data[3] = data[3].split('\n')[0]
            node_visted[data[0]] = int(-10)
            node_visted[data[1]] = int(-10)
            if data[0] not in graph:
                graph[data[0]] = dict()
            if data[1] not in graph:
                graph[data[1]] = dict()
            graph [data[0]] [data[1]] = (float(data[2]), float(data[3]))
    """
        3. check the end node to choose the data we need
        4. Load all data in heuristic.csv { node id : straight line distance with end node}
    """
    stra_dist = dict()
    case = 0
    with open(heuristicFile) as f:
        for line in f:
            data = line.split(',')
            data[3] = data[3].split('\n')[0]
            if data[0] == 'node':
                for i in range(1, 4):
                    if data[i] == str(end):
                        case = i
                        break
                continue
            stra_dist[ data[0] ] = float(data[case])
    """
        5. init return values
    """
    path = list()
    dist = float(0.0)
    num_visted = int(0)
    """
        6. A* init
            - found, a bool value means we found end node or not
            - create a prioity queue (q) to let smallest g() + h() be the top
                   - g(): the distance which have walked
                   - h(): straight line distance between that node and end node
            - put the start node int to q
            - mark start node in node_visted table as (START, 0)
    """
    found = bool(0)
    q = queue.PriorityQueue()
    q.put( (0 + stra_dist[str(start)], str(start)) )
    node_visted[str(start)] = ('START', 0.0)
    """
        7. A*
            - termination condition: found end point or no other point can touch
            - put every node adjacent with the top node in prioity queue (q) as (g() + h(), node id)
            - mark the node adjacent with the top node in node_visted table as (the top node, g() + h())
            - if the node is marked, we need to check the g() + h() is smaller or not
    """
    while q.empty() == 0 and found == 0:
        now = q.get()
        num_visted += 1
        if now[1] == str(end):
            found = 1
            break
        for new in graph[now[1]]:
            weight = graph[now[1]][new][0] + now[0] + stra_dist[new] - stra_dist[now[1]]
            if node_visted[new] == -10:
                node_visted[new] = (now[1], weight)
                q.put( (weight, new) )
            elif weight < node_visted[new][1]:
                node_visted[new] = (now[1], weight)
                q.put( (weight, new) )
    """
        8. get the path from end node
            - the mark of every node in node_visted is the prev node of the path from start node to end node
            - g() + h() of node_visted[end] will equal to the sum of the distance of these paths
            - append these nodes to list
            - when we see the mark of start is finish
    """
    now = str(end)
    dist = node_visted[now][1]
    if found == 1:
        while node_visted[now][0] != 'START':
            path.append(int(now))
            now = node_visted[now][0]
    """
        9. append the start node
        10. reverse the list
        11. return
    """    
    path.append(start)
    path.reverse()
    return path, dist, num_visted
    # End your code (Part 4)


if __name__ == '__main__':
    path, dist, num_visited = astar(1718165260, 8513026827)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
