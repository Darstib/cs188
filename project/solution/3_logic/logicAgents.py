# logicAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
This file contains all of the agents that can be selected to control Pacman.  To
select an agent, use the '-p' option when running pacman.py.  Arguments can be
passed to your agent using '-a'.  For example, to load a LogicAgent that uses
logicPlan.positionLogicPlan, run the following command:

> python pacman.py -p LogicAgent -a fn=positionLogicPlan

Commands to invoke other planning methods can be found in the project
description.

You should NOT change code in this file

Good luck and happy planning!
"""

from game import Directions
from game import Agent
from game import Actions
from game import Grid
from graphicsUtils import *
import graphicsDisplay
import util
import time
import warnings
import logicPlan
import random

class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP

#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of logicPlan.py       #
#######################################################

class LogicAgent(Agent):
    """
    This very general logic agent finds a path using a supplied planning
    algorithm for a supplied planning problem, then returns actions to follow that
    path.

    As a default, this agent runs positionLogicPlan on a
    PositionPlanningProblem to find location (1,1)

    Options for fn include:
      positionLogicPlan or plp
      foodLogicPlan or flp
      foodGhostLogicPlan or fglp


    Note: You should NOT change any code in LogicAgent
    """

    def __init__(self, fn='positionLogicPlan', prob='PositionPlanningProblem', plan_mod=logicPlan):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the planning function from the name and heuristic
        if fn not in dir(plan_mod):
            raise AttributeError(fn + ' is not a planning function in logicPlan.py.')
        func = getattr(plan_mod, fn)
        self.planningFunction = lambda x: func(x)

        # Get the planning problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError(prob + ' is not a planning problem type in logicAgents.py.')
        self.planType = globals()[prob]
        self.live_checking = False
        print('[LogicAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.planningFunction == None:
            raise Exception("No planning function provided for LogicAgent")
        starttime = time.time()
        problem = self.planType(state) # Makes a new planning problem

        self.actions = [] # In case planningFunction times out
        self.actions  = self.planningFunction(problem) # Find a path
        if self.actions == None:
            raise Exception('Studenct code supplied None instead of result')
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        # TODO Drop
        if '_expanded' in dir(problem):
            print('Nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        # import ipdb; ipdb.set_trace()
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            print('Oh no! The Pacman agent created a plan that was too short!')
            print()
            return None
            # return Directions.STOP

class CheckSatisfiabilityAgent(LogicAgent):
    def __init__(self, fn='checkLocationSatisfiability', prob='LocMapProblem', plan_mod=logicPlan):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the planning function from the name and heuristic
        if fn not in dir(plan_mod):
            raise AttributeError(fn + ' is not a planning function in logicPlan.py.')
        func = getattr(plan_mod, fn)
        self.planningFunction = lambda x: func(*x)

        # Get the planning problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError(prob + ' is not a planning problem type in logicAgents.py.')
        self.planType = globals()[prob]
        print('[LogicAgent] using problem type ' + prob)
        self.live_checking = False

    def registerInitialState(self, state):
        if self.planningFunction == None:
            raise Exception("No planning function provided for LogicAgent")
        starttime = time.time()
        self.problem = self.planType(state) # Makes a new planning problem

    def getAction(self, state):
        return "EndGame"

class LocalizeMapAgent(LogicAgent):
    """Parent class for localization, mapping, and slam"""
    def __init__(self, fn='positionLogicPlan', prob='LocMapProblem', plan_mod=logicPlan, display=None, scripted_actions=[]):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the planning function from the name and heuristic
        if fn not in dir(plan_mod):
            raise AttributeError(fn + ' is not a planning function in logicPlan.py.')
        func = getattr(plan_mod, fn)
        self.planningFunction = lambda x, y: func(x, y)

        # Get the planning problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError(prob + ' is not a planning problem type in logicAgents.py.')
        self.planType = globals()[prob]
        print('[LogicAgent] using problem type ' + prob)
        self.visited_states = []
        self.display = display
        self.scripted_actions = scripted_actions
        self.live_checking = True

    def resetLocation(self):
        self.visited_states = []
        self.state = self.problem.getStartState()
        self.visited_states.append(self.state)

    def addNoOp_t0(self):
        self.visited_states = [self.visited_states[0]] + list(self.visited_states)
        self.actions.insert(0, "Stop")

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.planningFunction == None:
            raise Exception("No planning function provided for LogicAgent")
        starttime = time.time()
        problem = self.planType(state) # Makes a new planning problem

        self.problem = problem
        self.state = self.problem.getStartState()

        self.actions = self.scripted_actions
        self.resetLocation()
        self.planning_fn_output = self.planningFunction(problem, self)
        # self.addNoOp_t0()

    def get_known_walls_non_walls_from_known_map(self, known_map):
        # map is 1 for known wall, 0 for 
        if known_map == None:
            raise Exception('Student code supplied None instead of a 2D known map')
        known_walls = [[(True if entry==1 else False) for entry in row] for row in known_map]
        known_non_walls = [[(True if entry==0 else False) for entry in row] for row in known_map]
        return known_walls, known_non_walls

class LocalizationLogicAgent(LocalizeMapAgent):
    def __init__(self, fn='localization', prob='LocalizationProblem', plan_mod=logicPlan, display=None, scripted_actions=[]):
        super(LocalizationLogicAgent, self).__init__(fn, prob, plan_mod, display, scripted_actions)
        self.num_timesteps = len(scripted_actions) if scripted_actions else 5

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        # import ipdb; ipdb.set_trace()
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1

        planning_fn_output = None
        if i < self.num_timesteps:
            proposed_action = self.actions[i]
            planning_fn_output = next(self.planning_fn_output)
            if planning_fn_output == None:
                raise Exception('Studenct code supplied None instead of result')
            if isinstance(self.display, graphicsDisplay.PacmanGraphics):
                self.drawPossibleStates(planning_fn_output, direction=self.actions[i])
        elif i < len(self.actions):
            proposed_action = self.actions[i]
        else:
            proposed_action = "EndGame"

        return proposed_action, planning_fn_output

    def moveToNextState(self, action):
        oldX, oldY = self.state
        dx, dy = Actions.directionToVector(action)
        x, y = int(oldX + dx), int(oldY + dy)
        if self.problem.walls[x][y]:
            raise AssertionError("Taking an action that goes into wall")
            pass
        else:
            self.state = (x, y)
        self.visited_states.append(self.state)

    def getPercepts(self):
        x, y = self.state
        north_iswall = self.problem.walls[x][y+1]
        south_iswall = self.problem.walls[x][y-1]
        east_iswall = self.problem.walls[x+1][y]
        west_iswall = self.problem.walls[x-1][y]
        return [north_iswall, south_iswall, east_iswall, west_iswall]

    def getValidActions(self):
        x, y = self.state
        actions = []
        if not self.problem.walls[x][y+1]: actions.append('North')
        if not self.problem.walls[x][y-1]: actions.append('South')
        if not self.problem.walls[x+1][y]: actions.append('East')
        if not self.problem.walls[x-1][y]: actions.append('West')
        return actions

    def drawPossibleStates(self, possibleLocations=None, direction="North", pacman_position=None):
        import __main__
        self.display.clearExpandedCells() # Erase previous colors
        self.display.colorCircleCells(possibleLocations, direction=direction, pacman_position=pacman_position)

class MappingLogicAgent(LocalizeMapAgent):
    def __init__(self, fn='mapping', prob='MappingProblem', plan_mod=logicPlan, display=None, scripted_actions=[]):
        super(MappingLogicAgent, self).__init__(fn, prob, plan_mod, display, scripted_actions)
        self.num_timesteps = len(scripted_actions) if scripted_actions else 10

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1

        planning_fn_output = None
        if i < self.num_timesteps:
            proposed_action = self.actions[i]
            planning_fn_output = next(self.planning_fn_output)
            if isinstance(self.display, graphicsDisplay.PacmanGraphics):
                self.drawWallBeliefs(planning_fn_output, self.actions[i], self.visited_states[:i])
        elif i < len(self.actions):
            proposed_action = self.actions[i]
        else:
            proposed_action = "EndGame"

        return proposed_action, planning_fn_output

    def moveToNextState(self, action):
        oldX, oldY = self.state
        dx, dy = Actions.directionToVector(action)
        x, y = int(oldX + dx), int(oldY + dy)
        if self.problem.walls[x][y]:
            raise AssertionError("Taking an action that goes into wall")
            pass
        else:
            self.state = (x, y)
        self.visited_states.append(self.state)

    def getPercepts(self):
        x, y = self.state
        north_iswall = self.problem.walls[x][y+1]
        south_iswall = self.problem.walls[x][y-1]
        east_iswall = self.problem.walls[x+1][y]
        west_iswall = self.problem.walls[x-1][y]
        return [north_iswall, south_iswall, east_iswall, west_iswall]

    def getValidActions(self):
        x, y = self.state
        actions = []
        if not self.problem.walls[x][y+1]: actions.append('North')
        if not self.problem.walls[x][y-1]: actions.append('South')
        if not self.problem.walls[x+1][y]: actions.append('East')
        if not self.problem.walls[x-1][y]: actions.append('West')
        return actions

    def drawWallBeliefs(self, known_map=None, direction="North", visited_states_to_render=[]):
        import random
        import __main__
        from graphicsUtils import draw_background, refresh
        known_walls, known_non_walls = self.get_known_walls_non_walls_from_known_map(known_map)
        wallGrid = Grid(self.problem.walls.width, self.problem.walls.height, initialValue=False)
        wallGrid.data = known_walls
        allTrueWallGrid = Grid(self.problem.walls.width, self.problem.walls.height, initialValue=True)
        self.display.clearExpandedCells() # Erase previous colors
        self.display.drawWalls(wallGrid, formatColor(.9,0,0), allTrueWallGrid)
        refresh()

class SLAMLogicAgent(LocalizeMapAgent):
    def __init__(self, fn='slam', prob='SLAMProblem', plan_mod=logicPlan, display=None, scripted_actions=[]):
        super(SLAMLogicAgent, self).__init__(fn, prob, plan_mod, display, scripted_actions)
        self.scripted_actions = scripted_actions
        self.num_timesteps = len(self.scripted_actions) if self.scripted_actions else 10
        self.live_checking = True

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        # import ipdb; ipdb.set_trace()
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        pacman_loc = self.visited_states[i]

        planning_fn_output = None
        if i < self.num_timesteps:
            proposed_action = self.actions[i]
            planning_fn_output = next(self.planning_fn_output)
            if planning_fn_output == None:
                raise Exception('Studenct code supplied None instead of result')
            if isinstance(self.display, graphicsDisplay.PacmanGraphics):
                self.drawWallandPositionBeliefs(
                    known_map=planning_fn_output[0],
                    possibleLocations=planning_fn_output[1],
                    direction=self.actions[i])
        elif i < len(self.actions):
            proposed_action = self.actions[i]
        else:
            proposed_action = "EndGame"

        # SLAM needs to handle illegal actions
        if proposed_action not in self.getValidActions(pacman_loc) and proposed_action not in ["Stop", "EndGame"]:
            proposed_action = "Stop"

        return proposed_action, planning_fn_output

    def moveToNextState(self, action):
        oldX, oldY = self.state
        dx, dy = Actions.directionToVector(action)
        x, y = int(oldX + dx), int(oldY + dy)
        if self.problem.walls[x][y]:
            # raise AssertionError("Taking an action that goes into wall")
            pass
        else:
            self.state = (x, y)
        self.visited_states.append(self.state)

    def getPercepts(self):
        x, y = self.state
        north_iswall = self.problem.walls[x][y+1]
        south_iswall = self.problem.walls[x][y-1]
        east_iswall = self.problem.walls[x+1][y]
        west_iswall = self.problem.walls[x-1][y]
        num_adj_walls = sum([north_iswall, south_iswall, east_iswall, west_iswall])
        # percept format: [adj_to_>=1_wall, adj_to_>=2_wall, adj_to_>=3_wall]
        percept = [num_adj_walls >= i for i in range(1, 4)]
        return percept

    def getValidActions(self, state=None):
        if not state:
            state = self.state
        x, y = state
        actions = []
        if not self.problem.walls[x][y+1]: actions.append('North')
        if not self.problem.walls[x][y-1]: actions.append('South')
        if not self.problem.walls[x+1][y]: actions.append('East')
        if not self.problem.walls[x-1][y]: actions.append('West')
        return actions

    def drawWallandPositionBeliefs(self, known_map=None, possibleLocations=None,
            direction="North", visited_states_to_render=[], pacman_position=None):
        import random
        import __main__
        from graphicsUtils import draw_background, refresh
        known_walls, known_non_walls = self.get_known_walls_non_walls_from_known_map(known_map)
        wallGrid = Grid(self.problem.walls.width, self.problem.walls.height, initialValue=False)
        wallGrid.data = known_walls
        allTrueWallGrid = Grid(self.problem.walls.width, self.problem.walls.height, initialValue=True)

        # Recover list of non-wall coords:
        non_wall_coords = []
        for x in range(len(known_non_walls)):
            for y in range(len(known_non_walls[x])):
                if known_non_walls[x][y] == 1:
                    non_wall_coords.append((x, y))

        self.display.clearExpandedCells() # Erase previous colors

        self.display.drawWalls(wallGrid, formatColor(.9,0,0), allTrueWallGrid)
        self.display.colorCircleSquareCells(possibleLocations, square_cells=non_wall_coords, direction=direction, pacman_position=pacman_position)
        refresh()

class PositionPlanningProblem(logicPlan.PlanningProblem):
    """
    A planning problem defines the state space, start state, goal test, successor
    function and cost function.  This planning problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this planning problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a planning state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print('Warning: this does not look like a regular position planning maze')

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def getGoalState(self):
        return self.goal

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999. 

        This is included in the logic project solely for autograding purposes.
        You should not be calling it.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost
    
    def getWidth(self):
        """
        Returns the width of the playable grid (does not include the external wall)
        Possible x positions for agents will be in range [1,width]
        """
        return self.walls.width-2

    def getHeight(self):
        """
        Returns the height of the playable grid (does not include the external wall)
        Possible y positions for agents will be in range [1,height]
        """
        return self.walls.height-2

def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionPlanningProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionPlanningProblem"
    xy1 = position
    xy2 = problem.goal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

class LocMapProblem:
    """Parent class for Localization, Mapping, and SLAM."""
    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def getWidth(self):
        """
        Returns the width of the playable grid (does not include the external wall)
        Possible x positions for agents will be in range [1,width]
        """
        return self.walls.width-2

    def getHeight(self):
        """
        Returns the height of the playable grid (does not include the external wall)
        Possible y positions for agents will be in range [1,height]
        """
        return self.walls.height-2

class LocalizationProblem(LocMapProblem):
    pass

class MappingProblem(LocMapProblem):
    pass

class SLAMProblem(LocMapProblem):
    pass

class FoodPlanningProblem:
    """
    A planning problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A planning state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """
    def __init__(self, startingGameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0 # DO NOT CHANGE
        self.heuristicInfo = {} # A dictionary for the heuristic to store information

    def getStartState(self):
        return self.start

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999. 

        This is included in the logic project solely for autograding purposes.
        You should not be calling it.
        """
        x,y= self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost
    
    def getWidth(self):
        """
        Returns the width of the playable grid (does not include the external wall)
        Possible x positions for agents will be in range [1,width]
        """
        return self.walls.width-2

    def getHeight(self):
        """
        Returns the height of the playable grid (does not include the external wall)
        Possible y positions for agents will be in range [1,height]
        """
        return self.walls.height-2
