HeapQ test
==================

```python
import heapq

# 빈 힙 생성
heap = []

# 힙에 요소 추가
heapq.heappush(heap, 5)
heapq.heappush(heap, 1)
heapq.heappush(heap, 3)

# 힙에서 가장 작은 요소 제거 및 반환
smallest = heapq.heappop(heap)  # smallest는 1이 됩니다.

# 힙 상태 출력
print(heap)  # [3, 5]
```


예시로, 2D 격자형 길찾기 문제에서 유클리드 거리나 맨해튼 거리와 같은 함수가 휴리스틱 함수로 자주 사용됩니다.

A* with Heuristic
==================

A* 알고리즘에서 **휴리스틱 함수 (Heuristic Function)**는 현재 노드에서 목표 노드까지의 예상 비용을 추정하는 데 사용됩니다. 
이 함수는 알고리즘이 더 효율적으로 목표에 도달하도록 도와주며, 길찾기 문제에서 최적의 경로를 찾는 데 중요한 역할을 합니다.

A* 알고리즘은 경로의 **총 비용 f(n)**을 최소화하는 방식으로 작동합니다. 여기서 f(n)은 다음과 같이 정의됩니다:

f(n)=g(n)+h(n)
---------------
    g(n): 시작 노드에서 현재 노드 nn까지 도달하는 실제 비용입니다.
    h(n): 현재 노드 nn에서 목표 노드까지의 예상 비용(휴리스틱 값)입니다.

휴리스틱 함수 h(n)가 어떻게 정의되느냐에 따라 A 알고리즘의 효율성과 정확성이 달라집니다. 보통 다음과 같은 조건을 만족해야 합니다:

    비관적인 추정 (Admissible): 휴리스틱 함수 h(n)h(n)은 실제 비용을 초과하지 않아야 합니다. 
    -----------------------------------------------------------------------------
    즉, 목표까지의 비용이 아무리 작아도 h(n)h(n)이 그보다 작거나 같아야 합니다. 이렇게 하면 A* 알고리즘이 항상 최적의 해를 찾을 수 있습니다.

    일관성 (Consistent 또는 Monotonic): 모든 노드 nn과 그 인접 노드 mm에 대해, 다음 조건이 성립해야 합니다.
    -------------------------------------------------------------------------------------------
    h(n)≤c(n,m)+h(m)

여기서 c(n,m)c(n,m)은 nn에서 mm으로 이동하는 실제 비용입니다. 일관성을 만족하는 휴리스틱 함수는 비관적인 특성을 자동으로 만족하게 되며, A* 알고리즘이 효율적으로 동작하게 해줍니다.


```python
# -----------
# Problem:
#
# A* 탐색 알고리즘을 사용하여 주어진 grid에서 목표 위치까지의 최적 경로를 찾고,
# 각 셀의 확장 순서를 포함한 expand 그리드를 반환하는 search() 함수입니다.
# 확장되지 않은 셀은 -1로 표시됩니다.
# 만약 init에서 goal까지 경로가 없다면 함수는 "Fail" 문자열을 반환해야 합니다.
# ----------

# 그리드와 휴리스틱 정보
grid = [[0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0]]

heuristic = [[9, 8, 7, 6, 5, 4],
             [8, 7, 6, 5, 4, 3],
             [7, 6, 5, 4, 3, 2],
             [6, 5, 4, 3, 2, 1],
             [5, 4, 3, 2, 1, 0]]

# 초기 위치와 목표 위치 정의
init = [0, 0]
goal = [len(grid) - 1, len(grid[0]) - 1]
cost = 1  # 이동 비용
delta = [[-1, 0],  # 위로 이동
         [0, -1],  # 왼쪽으로 이동
         [1, 0],   # 아래로 이동
         [0, 1]]   # 오른쪽으로 이동
delta_name = ['^', '<', 'v', '>']  # 이동 방향 이름

def search(grid, init, goal, cost, heuristic):
    # 각 위치가 방문되었는지 기록하는 'closed' 배열 (방문 시 1로 설정)
    closed = [[0 for col in range(len(grid[0]))] for row in range(len(grid))]
    closed[init[0]][init[1]] = 1  # 초기 위치 방문 처리
    
    # 확장 순서를 기록할 'expand' 배열, 초기에는 -1로 설정
    expand = [[-1 for col in range(len(grid[0]))] for row in range(len(grid))]
    
    # 이동 경로 기록용 action 배열, 초기에는 -1로 설정
    action = [[-1 for col in range(len(grid[0]))] for row in range(len(grid))]
    
    # 현재 위치 설정
    x = init[0]
    y = init[1]
    g = 0  # 시작 위치의 경로 비용
    f = g + heuristic[x][y]  # f값은 g와 heuristic의 합
    open = [[f, g, x, y]]  # 탐색할 셀을 [f, g, x, y] 형식으로 open 리스트에 추가
    
    found = False  # 목표 위치를 찾으면 True로 변경되는 플래그
    resign = False  # 경로를 찾을 수 없을 경우 True로 변경되는 플래그
    count = 0  # 확장된 셀의 순서를 기록하는 변수
    
    # 목표를 찾거나 탐색을 종료할 때까지 반복
    while not found and not resign:
        if len(open) == 0:  # open 리스트가 비어 있으면 경로가 없다고 판단
            resign = True
            return "Fail"
        else:
            # open 리스트에서 가장 작은 f 값을 가진 항목을 선택하고 삭제
            open.sort()
            open.reverse()
            next = open.pop()
            
            # 선택한 항목의 g, x, y 값을 가져옴
            g = next[1]
            x = next[2]
            y = next[3]
            f = next[0]
            
            # 현재 위치를 expand 배열에 확장된 순서로 기록
            expand[x][y] = count
            count += 1  # 확장 순서 증가
            
            # 목표 위치에 도달한 경우
            if x == goal[0] and y == goal[1]:
                found = True
            else:
                # 네 방향으로 이동 시도
                for i in range(len(delta)):
                    x2 = x + delta[i][0]
                    y2 = y + delta[i][1]
                    
                    # 이동할 위치가 그리드 안에 있는지 확인
                    if x2 >= 0 and x2 < len(grid) and y2 >= 0 and y2 < len(grid[0]):
                        # 이동 위치가 방문되지 않았고 장애물이 아닌 경우
                        if closed[x2][y2] == 0 and grid[x2][y2] == 0:
                            g2 = g + cost  # 새로운 g값
                            f2 = g2 + heuristic[x2][y2]  # 새로운 f값
                            open.append([f2, g2, x2, y2])  # open 리스트에 추가
                            closed[x2][y2] = 1  # 방문 처리
                            print(next)  # 현재 상태 출력 (디버깅용)
    
    return expand  # 확장된 순서를 포함한 expand 배열 반환

# search 함수 실행 및 결과 출력
print(search(grid, init, goal, cost, heuristic))

""" [0, -1, -1, -1, -1, -1], 
    [1, -1, -1, -1, -1, -1], 
    [2, -1, -1, -1, -1, -1], 
    [3, -1,  8,  9, 10, 11], 
    [4,  5,  6,  7, -1, 12] """
```




dijkstra Algorithm 
==================
A* 알고리즘에서 휴리스틱만 빠진 형태

비용이 최소인 경로를 찾는 데 필요한 정보만을 사용한다. 다익스트라는 모든 경로의 **실제 비용 g(n)g(n)**에만 의존하여 작동하며, 단순히 출발점에서 모든 노드로 가는 최단 경로를 구하는 데 최적화되어 있습니다. 따라서 목표 지점까지의 예상 비용(휴리스틱)을 추가로 고려하지 않아도 됩니다.
A* 알고리즘과 다익스트라 알고리즘의 차이

    다익스트라 알고리즘은 모든 경로의 실제 비용만 사용하기 때문에 균일한 비용 탐색을 수행합니다. 출발점에서 목표 지점까지의 모든 가능한 경로를 탐색하면서 비용이 가장 적게 드는 경로를 선택합니다. 휴리스틱이 없어서 계산 과정이 단순하며, 모든 경로를 평가하는 구조이기 때문에 목표 지점에 도달할 때까지 탐색이 필요해 시간이 많이 소요될 수 있습니다. 하지만 휴리스틱을 고려하지 않기에 최단 경로가 보장됩니다.

    *A 알고리즘**은 다익스트라 알고리즘과 달리 휴리스틱 함수를 추가로 사용하여 목표 지점까지의 예상 비용을 고려해 탐색을 효율적으로 합니다. 이로 인해 목표 지점에 가까운 경로를 우선적으로 탐색하게 되어 불필요한 경로를 덜 탐색하게 됩니다. 정확한 휴리스틱을 사용하면 최단 경로를 더 빨리 찾을 수 있지만, 휴리스틱 계산이 추가되므로 경우에 따라 다익스트라보다 계산이 복잡해질 수 있습니다.

휴리스틱 없는 방식이 더 빠를 때와 효율적인 경우

휴리스틱 없는 방식이 더 빠를 때는 다음과 같은 경우입니다:

    전체 탐색이 필요할 때: 목표 지점뿐 아니라 출발점에서 모든 지점에 대한 최단 경로를 구해야 할 경우, 다익스트라 알고리즘이 더 적합합니다.
    그래프가 작거나 단순한 경우: 목표 지점까지의 예측 비용을 계산하는 것이 오히려 오버헤드가 될 수 있습니다.

```python
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import heapq

# 그리드 설정 (1: 장애물 추가)
grid = np.array([
    [0, 0, 1, 0, 0, 0],
    [1, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 1, 0],
    [0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0]
])

# 시작 지점과 목표 지점 설정
start = (0, 0)
goal = (4, 5)

# 이동 방향 설정 (상, 하, 좌, 우)
delta = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def dijkstra(grid, start, goal):
    open_list = []
    heapq.heappush(open_list, (0, start))  # (비용, 위치)
    visited = set()
    visited_nodes = []
    cost_so_far = {start: 0}
    came_from = {}

    while open_list:
        current_cost, current = heapq.heappop(open_list)
        visited_nodes.append(current)

        if current == goal:
            break

        visited.add(current)

        for dx, dy in delta:
            neighbor = (current[0] + dx, current[1] + dy)

            if (0 <= neighbor[0] < grid.shape[0] and
                0 <= neighbor[1] < grid.shape[1] and
                grid[neighbor[0]][neighbor[1]] == 0 and
                neighbor not in visited):
                
                new_cost = current_cost + 1

                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost
                    heapq.heappush(open_list, (priority, neighbor))
                    came_from[neighbor] = current

    # 목표 노드까지의 경로 복원
    path = []
    if current == goal:
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()

    return path, visited_nodes

# 다익스트라 알고리즘 실행
optimal_path, visited_nodes = dijkstra(grid, start, goal)

# 시각화
fig, ax = plt.subplots()
ax.set_title('dijkstra algo')

def update(num):
    ax.clear()
    ax.imshow(grid, cmap='Greys')

    # 장애물 표시
    obstacles = np.argwhere(grid == 1)
    if obstacles.size > 0:
        x_obs, y_obs = zip(*obstacles)
        ax.plot(y_obs, x_obs, 's', color='black', markersize=10)

    # 방문한 노드 표시
    if num < len(visited_nodes):
        x_vals, y_vals = zip(*visited_nodes[:num+1])
        ax.plot(y_vals, x_vals, 'o', color='yellow', markersize=5)

    # 최적 경로 표시
    if optimal_path and num >= len(visited_nodes) - 1:
        x_opt, y_opt = zip(*optimal_path)
        ax.plot(y_opt, x_opt, color='blue', linewidth=2)

    # 시작 지점과 목표 지점 표시
    ax.plot(start[1], start[0], 's', color='green', markersize=10, label='시작')
    ax.plot(goal[1], goal[0], 's', color='red', markersize=10, label='목표')
    ax.legend()
    ax.set_xticks([])
    ax.set_yticks([])

ani = animation.FuncAnimation(fig, update, frames=len(visited_nodes)+10, interval=200, repeat=False)
plt.show()
```



JPS
==================
```python

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import heapq

# 큰 그리드 생성 (20x20)
np.random.seed(0)
grid_size = (20, 20)
grid = np.zeros(grid_size)
obstacle_probability = 0.2

for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        if np.random.rand() < obstacle_probability:
            grid[i][j] = 1
            
grid[0][0] = 0  # 시작 지점은 빈 공간으로 설정
grid[grid_size[0]-1][grid_size[1]-1] = 0  # 목표 지점은 빈 공간으로 설정

# 시작 지점과 목표 지점 설정
start = (0, 0)
goal = (grid_size[0]-1, grid_size[1]-1)

# 이동 방향 설정 (상, 하, 좌, 우, 대각선 포함)
delta = [(-1, 0), (1, 0), (0, -1), (0, 1),
         (-1, -1), (-1, 1), (1, -1), (1, 1)]

def heuristic(a, b):
    # 맨해튼 거리 사용
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def is_within_bounds(x, y):
    # 좌표가 그리드의 경계를 벗어나는지 확인하는 함수
    return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]

def jump(grid, current, direction):
    x, y = current
    dx, dy = direction
    while True:
        x, y = x + dx, y + dy
        if not is_within_bounds(x, y) or grid[x][y] == 1:
            return None
        if (x, y) == goal:
            return (x, y)
        
        # 직선 경로 상에서 강제 주변 점(Forced Neighbor)을 확인
        if dx != 0 and dy != 0:
            if (is_within_bounds(x - dx, y) and grid[x - dx][y] == 1 and grid[x][y - dy] == 0) or \
               (is_within_bounds(x, y - dy) and grid[x][y - dy] == 1 and grid[x - dx][y] == 0):
                return (x, y)
        elif dx != 0:
            if (is_within_bounds(x, y - 1) and grid[x][y - 1] == 1 and is_within_bounds(x, y + 1) and grid[x][y + 1] == 0) or \
               (is_within_bounds(x, y + 1) and grid[x][y + 1] == 1 and is_within_bounds(x, y - 1) and grid[x][y - 1] == 0):
                return (x, y)
        elif dy != 0:
            if (is_within_bounds(x - 1, y) and grid[x - 1][y] == 1 and is_within_bounds(x + 1, y) and grid[x + 1][y] == 0) or \
               (is_within_bounds(x + 1, y) and grid[x + 1][y] == 1 and is_within_bounds(x - 1, y) and grid[x - 1][y] == 0):
                return (x, y)
        # 점프가 계속 진행되도록 수정
        if dx != 0 or dy != 0:
            return (x, y)  # 강제 이웃을 찾지 못했어도 계속 진행

def jps(grid, start, goal):
    open_list = []
    heapq.heappush(open_list, (heuristic(start, goal), 0, start, [start]))
    visited = set()
    visited.add(start)
    visited_nodes = []
    
    while open_list:
        estimated_total, cost_so_far, current, path = heapq.heappop(open_list)
        visited_nodes.append(current)
        
        if current == goal:
            return path, visited_nodes
        
        for direction in delta:
            jump_point = jump(grid, current, direction)
            if jump_point and jump_point not in visited:
                visited.add(jump_point)
                total_cost = cost_so_far + heuristic(current, jump_point)
                estimated_total = total_cost + heuristic(jump_point, goal)
                heapq.heappush(open_list, (estimated_total, total_cost, jump_point, path + [jump_point]))
                
    return None, visited_nodes

# JPS+ 알고리즘 실행
optimal_path, visited_nodes = jps(grid, start, goal)

# 시각화
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_title('JPS+ 경로 탐색')

def update(num):
    ax.clear()
    ax.imshow(grid, cmap='Greys')
    # 장애물 표시
    obstacles = np.argwhere(grid == 1)
    if obstacles.size > 0:
        x_obs, y_obs = zip(*obstacles)
        ax.plot(y_obs, x_obs, 's', color='black', markersize=4)
    # 방문한 노드 표시
    if visited_nodes:
        x_vals, y_vals = zip(*visited_nodes[:num+1])
        ax.plot(y_vals, x_vals, 'o', color='yellow', markersize=2)
    # 최적 경로 표시
    if optimal_path and num >= len(visited_nodes) - 1:
        x_opt, y_opt = zip(*optimal_path)
        ax.plot(y_opt, x_opt, color='blue', linewidth=2)
    # 시작 지점과 목표 지점 표시
    ax.plot(start[1], start[0], 's', color='green', markersize=8, label='시작')
    ax.plot(goal[1], goal[0], 's', color='red', markersize=8, label='목표')
    ax.legend()
    ax.set_xticks([])
    ax.set_yticks([])

ani = animation.FuncAnimation(fig, update, frames=len(visited_nodes)+20, interval=50, repeat=False)
plt.show()

```



