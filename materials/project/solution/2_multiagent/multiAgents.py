# multiAgents.py
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # return successorGameState.getScore()
        "*** YOUR CODE HERE ***"
        # Get the food list and initial score
        foodList = newFood.asList()
        score = successorGameState.getScore()
        for food in foodList:
            score += 1 / util.manhattanDistance(newPos, food)

        # Ghost evaluation
        for i, ghostState in enumerate(newGhostStates):
            ghostPos = ghostState.getPosition()
            distance = util.manhattanDistance(newPos, ghostPos)
            if newScaredTimes[i] > 0:
                # Ghost is scared, it's good to be close
                score += 2 / (distance + 1)
            else:
                # Ghost is not scared, avoid it
                if distance < 2:
                    score -= 500  # Big penalty for being too close
                else:
                    score -= 2 / (distance + 1)

        return score


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth and self.evaluationFunction.
        Here are some method calls that might be useful when implementing minimax.
        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1
        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action
        gameState.getNumAgents():
        Returns the total number of agents in the game
        gameState.isWin():
        Returns whether or not the game state is a winning state
        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def getValue(state, agentIndex, level):
            agentIndex = agentIndex % state.getNumAgents()
            if (
                state.isWin()
                or state.isLose()
                or level == self.depth * gameState.getNumAgents() - 1
            ):
                return self.evaluationFunction(state)
            elif agentIndex == 0:
                return max(
                    getValue(
                        state.generateSuccessor(agentIndex, action),
                        agentIndex + 1,
                        level + 1,
                    )
                    for action in state.getLegalActions(agentIndex)
                )
            else:
                return min(
                    getValue(
                        state.generateSuccessor(agentIndex, action),
                        agentIndex + 1,
                        level + 1,
                    )
                    for action in state.getLegalActions(agentIndex)
                )

        # Pacman is always agent 0, and the agents move in order of increasing agent index.
        legalActions = gameState.getLegalActions(0)
        scores = [
            getValue(
                gameState.generateSuccessor(0, action),
                1,
                0,
            )
            for action in legalActions
        ]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
        return legalActions[chosenIndex]

    "参考：https://github.com/szzxljr/CS188_Course_Projects/blob/master/proj2multiagent/multiAgents.py#L140"
    # def getAction(self, gameState):
    #     def cal_value(state, numsOfagent, depth, f, score):
    #         legal_moves = state.getLegalActions(numsOfagent)
    #         for action in legal_moves:
    #             next_state = state.generateSuccessor(numsOfagent, action)
    #             score = f(score, get_value(next_state, numsOfagent + 1, depth - 1))
    #         return score

    #     def get_value(state, numsOfagent, depth):
    #         numsOfagent = numsOfagent % state.getNumAgents()
    #         if state.isWin() or state.isLose() or depth == 0:
    #             return self.evaluationFunction(state)
    #         elif numsOfagent == 0:
    #             return cal_value(state, numsOfagent, depth, max, -1000000)
    #         else:
    #             return cal_value(state, numsOfagent, depth, min, 1000000)

    #     scores = [
    #         get_value(
    #             gameState.generateSuccessor(0, action),
    #             1,
    #             self.depth * gameState.getNumAgents() - 1,
    #         )
    #         for action in gameState.getLegalActions(0)
    #     ]
    #     best_score = max(scores)
    #     best_indices = [
    #         index for index in range(len(scores)) if scores[index] == best_score
    #     ]
    #     index = random.choice(best_indices)
    #     return gameState.getLegalActions(0)[index]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def getValue(state, agentIndex, level, alpha, beta):
            agentIndex = agentIndex % state.getNumAgents()
            if (
                state.isWin()
                or state.isLose()
                or level == self.depth * state.getNumAgents()
            ):
                return self.evaluationFunction(state)
            elif agentIndex == 0:
                return max_value(state, agentIndex, level, alpha, beta)
            else:
                return min_value(state, agentIndex, level, alpha, beta)

        def max_value(state, agentIndex, level, alpha, beta):
            v = float("-inf")
            for action in state.getLegalActions(agentIndex):
                v = max(
                    v,
                    getValue(
                        state.generateSuccessor(agentIndex, action),
                        agentIndex + 1,
                        level + 1,
                        alpha,
                        beta,
                    ),
                )
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v

        def min_value(state, agentIndex, level, alpha, beta):
            v = float("inf")
            for action in state.getLegalActions(agentIndex):
                v = min(
                    v,
                    getValue(
                        state.generateSuccessor(agentIndex, action),
                        agentIndex + 1,
                        level + 1,
                        alpha,
                        beta,
                    ),
                )
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        # Pacman is always agent 0, and the agents move in order of increasing agent index.
        legalActions = gameState.getLegalActions(0)
        alpha = float("-inf")
        beta = float("inf")
        bestScore = float("-inf")
        bestAction = None

        for action in legalActions:
            score = getValue(gameState.generateSuccessor(0, action), 1, 1, alpha, beta)
            if score > bestScore:
                bestScore = score
                bestAction = action
            alpha = max(alpha, bestScore)

        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def getValue(state, agentIndex, level):
            agentIndex = agentIndex % state.getNumAgents()
            if (
                state.isWin()
                or state.isLose()
                or level == self.depth * gameState.getNumAgents() - 1
            ):
                return self.evaluationFunction(state)
            elif agentIndex == 0:
                return max(
                    getValue(
                        state.generateSuccessor(agentIndex, action),
                        agentIndex + 1,
                        level + 1,
                    )
                    for action in state.getLegalActions(agentIndex)
                )
            else:
                return sum(
                    getValue(
                        state.generateSuccessor(agentIndex, action),
                        agentIndex + 1,
                        level + 1,
                    )
                    for action in state.getLegalActions(agentIndex)
                ) / len(state.getLegalActions(agentIndex))

        # Pacman is always agent 0, and the agents move in order of increasing agent index.
        legalActions = gameState.getLegalActions(0)
        scores = [
            getValue(
                gameState.generateSuccessor(0, action),
                1,
                0,
            )
            for action in legalActions
        ]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best
        return legalActions[chosenIndex]


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    Just as what we do in ReflexAgent, but now we evaluate currentGameState
    """
    "*** YOUR CODE HERE ***"
    Pos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
    foodList = Food.asList()
    score = currentGameState.getScore()
    for food in foodList:
        score += 1 / util.manhattanDistance(Pos, food)
    # Ghost evaluation
    for i, ghostState in enumerate(GhostStates):
        ghostPos = ghostState.getPosition()
        distance = util.manhattanDistance(Pos, ghostPos)
        if ScaredTimes[i] > 0:
            # Ghost is scared, it's good to be close
            score += 2 / (distance + 1)
        else:
            # Ghost is not scared, avoid it
            if distance < 2:
                score -= 500  # Big penalty for being too close
            else:
                score -= 2 / (distance + 1)
    return score
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
