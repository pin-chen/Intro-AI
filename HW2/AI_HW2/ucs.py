import csv
#
import queue
#
edgeFile = 'edges.csv'


def ucs(start, end):
    # Begin your code (Part 3)
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
            graph[data[0]][data[1]] = (float(data[2]), float(data[3]))
    """
        3. init return values
    """
    path = list()
    dist = float(0.0)
    num_visted = int(0)
    """
        4. UCS init
            - found, a bool value means we found end node or not
            - create a prioity queue (q) to let smallest distance be the top
            - put the start node int to q
            - mark start node in node_visted table as (START, 0)
    """
    found = bool(0)
    q = queue.PriorityQueue()
    q.put( (0, str(start)) )
    node_visted[str(start)] = ('START', 0.0)
    """
        5. UCS
            - termination condition: found end point or no other point can touch
            - put every node adjacent with the top node in prioity queue (q) as (the distance have walked, node id)
            - mark the node adjacent with the top node in node_visted table as (the top node, the distance have walked)
            - if the node is marked, we need to check the the distance have walked is smaller or not
    """
    while q.empty() == 0 and found == 0:
        now = q.get()
        num_visted += 1
        if now[1] == str(end):
            found = 1
            break
        for new in graph[now[1]]:
            weight =  now[0] + graph[now[1]][new][0]
            if node_visted[new] == -10:
                node_visted[new] = (now[1], weight)
                q.put( (weight, new) )
            elif weight < node_visted[new][1]:
                node_visted[new] = (now[1], weight)
                q.put( (weight, new) )
    """
        6. get the path from end node
            - the mark of every node in node_visted is the prev node of the path from start node to end node
            - the distance of node_visted[end] will equal to the sum of the distance of these paths
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
        7. append the start node
        8. reverse the list
        9. return
    """ 
    path.append(start)
    path.reverse()
    return path, dist, num_visted
    # End your code (Part 3)


if __name__ == '__main__':
    path, dist, num_visited = ucs(426882161, 1737223506)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
