import csv
#
import queue
#
edgeFile = 'edges.csv'


def bfs(start, end):
    # Begin your code (Part 1)
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
        3. init return values
    """
    path = list()
    dist = float(0.0)
    num_visted = int(0)
    """
        4. BFS init
            - found, a bool value means we found end node or not
            - create a FIFO datastructure, q
            - put the start node int to q
            - mark start node in node_visted table as START
    """
    found = bool(0)
    q = queue.Queue()
    q.put( str(start) )
    node_visted[str(start)] = 'START'
    """
        5. BFS
            - termination condition: found end point or no other point can touch
            - put every node adjacent with the top node in queue
            - mark the node adjacent with the top node in node_visted table as the top node
    """
    while q.empty() == 0 and found == 0:
        now = q.get()
        num_visted += 1
        for new in graph[now]:
            if node_visted[new] == -10:
                node_visted[new] = now
                if new == str(end):
                    found = 1
                    break
                q.put(new)
    """
        6. get the path from end node
            - the mark of every node in node_visted is the prev node of the path from start node to end node
            - sum the distance of these paths
            - append these nodes to list
            - when we see the mark of start is finish
    """    
    now = str(end)
    if found == 1:
        while node_visted[now] != 'START':
            path.append(int(now))
            dist += graph[node_visted[now]][now][0]
            now = node_visted[now]
    """
        7. append the start node
        8. reverse the list
        9. return
    """
    path.append(start)
    path.reverse()
    return path, dist, num_visted
    # End your code (Part 1)


if __name__ == '__main__':
    path, dist, num_visited = bfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
