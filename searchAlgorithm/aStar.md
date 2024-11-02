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



