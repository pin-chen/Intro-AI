import csv
#
import queue
#
edgeFile = 'edges.csv'
heuristicFile = 'heuristic.csv'


def astar_time(start, end):
    # Begin your code (Part 6)
    #raise NotImplementedError("To be implemented")
    """
        different thing:
        1. To find the largest speed limit
        2. change the speed limit from km/hr to m/s
        others same with A*
    """
    graph = dict()
    node_visted = dict()
    Max = 0
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
            graph [data[0]] [data[1]] = (float(data[2]), float(data[3]) / 60 / 60 * 1000)
            Max = max(Max, float(data[3]) / 60 / 60 * 1000 )
    """
        same with A*
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
        same with A*
    """
    path = list()
    time = float(0.0)
    num_visted = int(0)
    """
        3. weight
            - g(): the fatest time we need to go to this node (distance / speed limit)
            - h(): straight line distance between that node and end node / the max speed limit of this graph
        same with A*
    """
    found = bool(0)
    q = queue.PriorityQueue()
    q.put( (0 + stra_dist[str(start)] / Max, str(start)) )
    node_visted[str(start)] = ('START', 0.0)
    """
        same with A*
    """
    while q.empty() == 0 and found == 0:
        now = q.get()
        num_visted += 1
        if now[1] == str(end):
            found = 1
            break
        for new in graph[now[1]]:
            weight = graph[now[1]][new][0] / graph[now[1]][new][1] + now[0] + stra_dist[new] / Max - stra_dist[now[1]] / Max
            if node_visted[new] == -10:
                node_visted[new] = (now[1], weight)
                q.put( (weight, new) )
            elif weight < node_visted[new][1]:
                node_visted[new] = (now[1], weight)
                q.put( (weight, new) )
    """
        same with A*
    """
    now = str(end)
    time = node_visted[now][1]
    if found == 1:
        while node_visted[now][0] != 'START':
            path.append(int(now))
            now = node_visted[now][0]
    """
        same with A*
    """     
    path.append(start)
    path.reverse()
    return path, time, num_visted
    # End your code (Part 6)


if __name__ == '__main__':
    path, time, num_visited = astar_time(1718165260, 8513026827)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total second of path: {time}')
    print(f'The number of visited nodes: {num_visited}')
