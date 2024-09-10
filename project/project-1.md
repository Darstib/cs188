---
tags:
  - notes
comments: true
dg-publish: true
---

> [!PREREQUISITE]
>
> - [02-State_Spaces_Uninformed_Search](../note/02-State_Spaces_Uninformed_Search.md)
> - [03-Informed_Search_Astar_and_Heuristics](../note/03-Informed_Search_Astar_and_Heuristics.md)
> - [preject 1](https://inst.eecs.berkeley.edu/~cs188/sp24/projects/proj1/) （若需要认证，可见[仓库](https://github.com/Darstib/cs188/tree/main/materials/project/intro_page)）

## explain

> 前面有很多探索，希望看正确解答可直接看 "right"；最终结果见 "pass" 。

pacman.py 的参数对于对于帮助我们进行测试很有用：

```shell title="man pacman"
$ python pacman.py -h

Usage: 
    USAGE:      python pacman.py <options>
    EXAMPLES:   (1) python pacman.py
                    - starts an interactive game
                (2) python pacman.py --layout smallClassic --zoom 2
                OR  python pacman.py -l smallClassic -z 2
                    - starts an interactive game on a smaller board, zoomed in
    

Options:
  -h, --help            show this help message and exit
  -n GAMES, --numGames=GAMES
                        the number of GAMES to play [Default: 1]
  -l LAYOUT_FILE, --layout=LAYOUT_FILE
                        the LAYOUT_FILE from which to load the map layout
                        [Default: mediumClassic]
  -p TYPE, --pacman=TYPE
                        the agent TYPE in the pacmanAgents module to use
                        [Default: KeyboardAgent]
  -t, --textGraphics    Display output as text only
  -q, --quietTextGraphics
                        Generate minimal output and no graphics
  -g TYPE, --ghosts=TYPE
                        the ghost agent TYPE in the ghostAgents module to use
                        [Default: RandomGhost]
  -k NUMGHOSTS, --numghosts=NUMGHOSTS
                        The maximum number of ghosts to use [Default: 4]
  -z ZOOM, --zoom=ZOOM  Zoom the size of the graphics window [Default: 1.0]
  -f, --fixRandomSeed   Fixes the random seed to always play the same game
  -r, --recordActions   Writes game histories to a file (named by the time
                        they were played)
  --replay=GAMETOREPLAY
                        A recorded game file (pickle) to replay
  -a AGENTARGS, --agentArgs=AGENTARGS
                        Comma separated values sent to agent. e.g.
                        "opt1=val1,opt2,opt3=val3"
  -x NUMTRAINING, --numTraining=NUMTRAINING
                        How many episodes are training (suppresses output)
                        [Default: 0]
  --frameTime=FRAMETIME
                        Time to delay between frames; <0 means keyboard
                        [Default: 0.1]
  -c, --catchExceptions
                        Turns on exception handling and timeouts during games
  --timeout=TIMEOUT     Maximum length of time an agent can spend computing in
                        a single game [Default: 30]
```

若加入下列语句，测试运行：

```python
print("Start:", problem.getStartState())
print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
print("Start's successors:", problem.getSuccessors(problem.getStartState()))

# Start: (5, 5)
# Is the start a goal? False
# Start's successors: [((5, 4), 'South', 1), ((4, 5), 'West', 1)]
```

窥见了 problem/state 的结构，其中 successors 是两个 successor，其内容依次代表：(state，action，cost) => （坐标，方向，步长）。

随便补全 dfs 观察[地图](attachments/project-1.png)：

```python title="dfs test"
def depthFirstSearch(problem: SearchProblem):
    return []
    util.raiseNotDefined()
```

可以看出来，“坐标原点”在左下角，是经典的“2D函数坐标”而非“矩阵坐标”。

### Q1 - Q4
#### explore

根据以前所学，dfs尝试 Q1：

```python title="dfs"
def depthFirstSearch(problem: SearchProblem):
    reached = set()  # 记录已经访问过的节点
    _stack = util.Stack()  # 用栈来实现DFS
    _stack.push(
        (problem.getStartState(), [])
    )  # 将初始状态（坐标，到达该坐标需要的方向组）压入栈中
    while not _stack.isEmpty():
        state, actions = _stack.pop()
        if problem.isGoalState(state):
            return actions
        if state not in reached:
            reached.add(state)
            for successor, action, _ in problem.getSuccessors(state):
                new_path = actions + [action]
                _stack.push((successor, new_path))
    return actions
    util.raiseNotDefined()
```

[顺利通过 Q1](attachments/project-1-1.png)。

#### right

下一个？NO! 在 note02 中我们知道三个搜索算法总体差不多，且这里都是 graph search:

> [!CITE]
>
> Now it’s time to write **full-fledged generic search** functions to help Pacman plan routes! Pseudocode for the search algorithms you’ll write can be found in the lecture slides.

```python title="pseudocode for graph search in note03"
function GRAPH-SEARCH(problem, frontier) return a solution or failure
    reached ← an empty set
    frontier ← INSERT(MAKE-NODE(INITIAL-STATE[problem]), frontier)
    while not IS-EMPTY(frontier) do
        node ← POP(frontier)
        if problem.IS-GOAL(node.STATE) then
            return node
        end
        if node.STATE is not in reached then
            add node.STATE in reached
            for each child-node in EXPAND(problem, node) do
                frontier ← INSERT(child-node, frontier)
            end
        end
    end
    return failure
```

所以我们照猫画虎，给出：

```python title="graphSearch"
def graphSearch(problem, frontier):
    """
    Implements the graph search algorithm using the provided frontier data structure.
    """
    reached = set()
    frontier.push((problem.getStartState(), []))  # take (state, actions) as node
    
    while not frontier.isEmpty():
        state, actions = frontier.pop()
    
        if problem.isGoalState(state):
            return actions
    
        if state not in reached:
            reached.add(state)
            for successor, action, _ in problem.getSuccessors(state):
                new_path = actions + [action]
                frontier.push((successor, new_path))
    
    return None  # No solution found
```

显然，这里的 frontier 在 DFS 中就是 Stack，在 BFS 中就是 Queue；在 UCS/ $A^*S$  中是 PriorityQueue。

前两个好说；PrigirityQueue 的 push 与前二者参数就不同，自然是报错了；但是注意到 util.py 中的：

```python title="PriorityQueueWithFunction"
class PriorityQueueWithFunction(PriorityQueue): # success PriorityQueue
    """
    Implements a priority queue with the same push/pop signature of the
    Queue and the Stack classes. This is designed for drop-in replacement for
    those two classes. The caller has to provide a priority function, which
    extracts each item's priority.
    """
    def  __init__(self, priorityFunction):
        "priorityFunction (item) -> priority"
        self.priorityFunction = priorityFunction      # store the priority function
        PriorityQueue.__init__(self)        # super-class initializer
    
    def push(self, item):
        "Adds an item to the queue with priority from the priority function"
        PriorityQueue.push(self, item, self.priorityFunction(item))
```

根据注释我们可以看出来这就是对 PriorityQueue 的封装了，唯一不同的是我们需要使用一个 priorityFunction(item) 来实例化它（按照 Stack 类中的使用，item 应该指我们设定的元组 node）；再看：

```python title="getCostOfActions"
class SearchProblem:
    ...
    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take
    
        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()
```

用这个方法获取 cost，作为 UCS minheap 的依据；而 $A^*S$ 多加了一个 heuristic value 即可；你说巧不巧，aStarSearch 的参数就有这么一个函数，[Q1 - Q4 通过](attachments/project-1-2.png) 。

### Q5 (3 pts): Finding All the Corners (Lecture 3)

> 完成这题使用了 bfs，所以需要先运行 `$ python pacman.py -l tinyCorners -p SearchAgent -a fn=bfs` 确保通过。

#### explore

这关实现一个 CornersProblem 类；首先我们看 `class CornersProblem(search.SearchProblem):` 其继承了我们之前 search.py 中的 SearchProblem 定好的模板，所以我们应当参考 Q1-Q4 所期望的功能来实现它；又在该文件中搜 `search.SearchProblem` ，找到了 `class PositionSearchProblem(search.SearchProblem):` 这个兄弟，还是实现好的；那么我们继续照猫画虎：

```python title="Corners Problem"
def getStartState(self):
    return self.startingPosition
    util.raiseNotDefined()

def isGoalState(self, state: Any):
    # 暂时不会，让其始终返回False
    util.raiseNotDefined()

def getSuccessors(self, state: Any):
    successors = []
    for action in [
        Directions.NORTH,
        Directions.SOUTH,
        Directions.EAST,
        Directions.WEST,
    ]:
        x, y = state
        dx, dy = Actions.directionToVector(action)
        nextx, nexty = int(x + dx), int(y + dy)
        if not self.walls[nextx][nexty]:
            nextState = (nextx, nexty)
            # cost = self.costFn(nextState)
            cost = 1
            successors.append((nextState, action, cost))
    self._expanded += 1  # DO NOT CHANGE
    return successors
```

我们让 IsGoalState 始终返回 False，那么 pacman 始终没有一条路径可以走，必然静止不动；运行 `$ python pacman.py -l tinyCorners -p SearchAgent -a fn=bfs,prob=CornersProblem` [看看](attachments/project-1-3.png)。

那么我们该怎么写 IsGoalState 呢？

#### right

> 下面内容参考 [szzxljr 的代码](https://github.com/szzxljr/CS188_Course_Projects/blob/7beea60ec9037fa5b750d211c90d814a954cbdfe/proj1search/searchAgents.py#L241)，其实也体现在后面的 FoodSearchProblem 中。

我们假想我们上述的 node 不仅带着位置信息和动作，还带着一个 corners；每经过一个对应的 corner，我们就丢掉这个 corner；当我们丢完时，就达到目标了；但是直接(state, actions, corners) 是不行的[^1]。所以我们将其和原 state 作为新元组，即 state <= (state, corners):

[^1]: 我们需要和之前实现的 BFS 接口，这个 node 应该只有两个成员。

```python title="right Corners Problem"
def getStartState(self):
    return (self.startingPosition, self.corners)
    util.raiseNotDefined()

def isGoalState(self, state: Any):
    return len(state[1]) == 0
    util.raiseNotDefined()

def getSuccessors(self, state: Any):
    successors = []
    for action in [
        Directions.NORTH,
        Directions.SOUTH,
        Directions.EAST,
        Directions.WEST,
    ]:
        x, y = state[0]
        corners = state[1]
        dx, dy = Actions.directionToVector(action)
        nextx, nexty = int(x + dx), int(y + dy)
        if not self.walls[nextx][nexty]:
            nextState0 = (nextx, nexty)
            corners = tuple(filter(lambda c: c != nextState0, corners))
            # cost = self.costFn(nextState0) # accoding to the hint, cost is 1
            cost = 1
            successors.append(((nextState0, corners), action, cost))
    # print(successors)
    self._expanded += 1  # DO NOT CHANGE
    return successors
```

[Q5通过](attachments/project-1-4.png)

### Q6-7 

#### explore

Q6 其实就是让我们想一个比较好的 `def cornersHeuristic(state: Any, problem: CornersProblem):` 函数；在数学中描述“距离”的[范数](https://www.wikiwand.com/zh/articles/%E8%8C%83%E6%95%B0)就比较适合先试试手；曼哈顿距离也是其中一种，且 SearchAgent.py 中有实现：

```python title="manhattanHeuristic"
def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])
```

我们可以计算此时相比于剩下的 corner 的曼哈顿距离，并取最小者即可：

```python title="cornerHeuristic using min manhattanDistance"
def manhattanDistance(m, n):
    return abs(m[0] - n[0]) + abs(m[1] - n[1])

def cornersHeuristic(state: Any, problem: CornersProblem):
    corners = problem.corners  # These are the corner coordinates
    walls = problem.walls  # These are the walls of the maze, as a Grid (game.py)
    if len(state[1]) == 0:
        return 0
    min_distance = float("inf")
    for corner in state[1]:
        tmp = manhattanDistance(corner, state[0])
        min_distance = min(min_distance, tmp)
    return min_distance
```

[Q6半通过](attachments/project-1-5.png)

#### right

但是想想，manhattenDistance 本身就一定不比实际需要的步数多，应当取最大值来贴近；同时发现 util.py 中实现了曼哈顿距离：

```python title="cornerHeuristic using max manhattanDistance"
def cornersHeuristic(state: Any, problem: CornersProblem):
    if len(state[1]) == 0:
        return 0
    max_distance = 0
    for corner in state[1]:
        # tmp = manhattanDistance(corner, state[0])
        tmp = util.manhattanDistance(corner, state[0])
        max_distance = max(max_distance, tmp)
    return max_distance
```

[Q6通过](attachments/project-1-6.png)

Q7 也是同理，之前是 4 个豆子在四个角落，现在是若干个豆子在某某个位置；所以思路基本一致（其中的 mazeDistance 是后来看了其他人的题解发现的）：

```python title=""
def foodHeuristic(state: Tuple[Tuple, List[List]], problem: FoodSearchProblem):
    position, foodGrid = state
    distances = [0]
    for food in foodGrid.asList():
        distances.append(mazeDistance(position, food, problem.startingGameState))
    return max(distances)
```

[Q7通过](attachments/project-1-7.png)

### Q8 (3 pts): Suboptimal Search

#### right 

根据提示，我们先补全 AnyFoodSearchProblem：

```python title="AnyFoodSearchProblem"
class AnyFoodSearchProblem(PositionSearchProblem):
    ...
    def isGoalState(self, state: Tuple[int, int]):
    x, y = state
    return self.food[x][y] # 去看这个成员的定义就懂了
    util.raiseNotDefined()
```

也就是说，这个类在此实现的功能就是检查某个位置食物是否被吃了。

Q8 本身更简单了，`Returns a path (a list of actions) to the closest dot, starting from gameState` ，在每一步成本都为 1 的情况下，最近既是成本最小，uniformCostSearch 即可：

```python title="findPathToClosestDot"
def findPathToClosestDot(self, gameState: pacman.GameState):
    """
    Returns a path (a list of actions) to the closest dot, starting from
    gameState.
    """
    # Here are some useful elements of the startState
    startPosition = gameState.getPacmanPosition()
    food = gameState.getFood()
    walls = gameState.getWalls()
    problem = AnyFoodSearchProblem(gameState)
    
    "*** YOUR CODE HERE ***"
    return search.uniformCostSearch(problem)
    util.raiseNotDefined()
```

[Q8 通过](attachments/project-1-8.png)
## pass

- [project-1 全部通过](attachments/project-1-9.png)
- [全代码](https://github.com/Darstib/cs188/tree/main/project/solution)

> [!SUMMARY]
>
> 多看相关注释和已有的类似实现；开始前清楚自己能用哪些东西。
