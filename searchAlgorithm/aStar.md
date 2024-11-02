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
