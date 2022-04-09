from util import manhattanDistance
from game import Directions
import random, util
from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        minGhostDistance = min([manhattanDistance(newPos, state.getPosition()) for state in newGhostStates])

        scoreDiff = childGameState.getScore() - currentGameState.getScore()

        pos = currentGameState.getPacmanPosition()
        nearestFoodDistance = min([manhattanDistance(pos, food) for food in currentGameState.getFood().asList()])
        newFoodsDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        newNearestFoodDistance = 0 if not newFoodsDistances else min(newFoodsDistances)
        isFoodNearer = nearestFoodDistance - newNearestFoodDistance
            
        direction = currentGameState.getPacmanState().getDirection()
        if minGhostDistance <= 1 or action == Directions.STOP:
            return 0
        if scoreDiff > 0:
            return 8
        elif isFoodNearer > 0:
            return 4
        elif action == direction:
            return 2
        else:
            return 1


def scoreEvaluationFunction(currentGameState):
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (Part 1)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        # Begin your code (Part 1)
        #raise NotImplementedError("To be implemented")
        _, nextAction = self.maxValue(gameState, 0, 0);
        return nextAction
    def terminalState(self, gameState, agentIndex, curDepth):
        if gameState.isWin() or gameState.isLose() or curDepth >= self.depth:
            return True
        return False
    def nextValue(self, gameState, agentIndex, curDepth, nextAgent, legalAction):
        nextState = gameState.getNextState(agentIndex, legalAction)
        legalValue = float()
        if nextAgent == 0:
            legalValue, _ = self.maxValue(nextState, nextAgent, curDepth + 1)
        else:
            legalValue = self.minValue(nextState, nextAgent, curDepth)
        return legalValue
    def maxValue(self, gameState, agentIndex, curDepth):
        if( self.terminalState(gameState, agentIndex, curDepth) ):
            return self.evaluationFunction(gameState), None
        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        scores = list()
        legalMoves = gameState.getLegalActions(agentIndex)
        for legalAction in legalMoves:
            legalValue = self.nextValue(gameState, agentIndex, curDepth, nextAgent, legalAction)
            scores.append(legalValue)
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)
        return bestScore, legalMoves[chosenIndex]
    def minValue(self, gameState, agentIndex, curDepth):
        if( self.terminalState(gameState, agentIndex, curDepth) ):
            return self.evaluationFunction(gameState)
        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        bestScore = float("inf")
        for legalAction in gameState.getLegalActions(agentIndex):
            legalValue = self.nextValue(gameState, agentIndex, curDepth, nextAgent, legalAction)
            bestScore = min(bestScore, legalValue)
        return bestScore
        # End your code (Part 1)
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (Part 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # Begin your code (Part 2)
        #raise NotImplementedError("To be implemented")
        _, nextAction = self.maxValue(gameState, 0, 0, float("-inf"), float("inf"));
        return nextAction
    def terminalState(self, gameState, agentIndex, curDepth):
        if gameState.isWin() or gameState.isLose() or curDepth >= self.depth:
            return True
        return False
    def nextValue(self, gameState, agentIndex, curDepth, nextAgent, legalAction, alpha, beta):
        nextState = gameState.getNextState(agentIndex, legalAction)
        legalValue = float()
        if nextAgent == 0:
            legalValue, _ = self.maxValue(nextState, nextAgent, curDepth + 1, alpha, beta)
        else:
            legalValue = self.minValue(nextState, nextAgent, curDepth, alpha, beta)
        return legalValue
    def maxValue(self, gameState, agentIndex, curDepth, alpha, beta):
        if( self.terminalState(gameState, agentIndex, curDepth) ):
            return self.evaluationFunction(gameState), None
        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        scores = list()
        legalMoves = gameState.getLegalActions(agentIndex)
        for legalAction in legalMoves:
            legalValue = self.nextValue(gameState, agentIndex, curDepth, nextAgent, legalAction, alpha, beta)
            scores.append(legalValue)
            if legalValue > beta:
                return legalValue, legalAction
            alpha = max(alpha, legalValue)
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)
        return bestScore, legalMoves[chosenIndex]
    def minValue(self, gameState, agentIndex, curDepth, alpha, beta):
        if( self.terminalState(gameState, agentIndex, curDepth) ):
            return self.evaluationFunction(gameState)
        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        bestScore = float("inf")
        for legalAction in gameState.getLegalActions(agentIndex):
            legalValue = self.nextValue(gameState, agentIndex, curDepth, nextAgent, legalAction, alpha, beta)
            bestScore = min(bestScore, legalValue)
            if legalValue < alpha:
                return legalValue
            beta = min(beta, bestScore)
        return bestScore
        # End your code (Part 2)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (Part 3)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        # Begin your code (Part 3)
        #raise NotImplementedError("To be implemented")
        _, nextAction = self.maxValue(gameState, 0, 0);
        return nextAction
    def terminalState(self, gameState, agentIndex, curDepth):
        if gameState.isWin() or gameState.isLose() or curDepth >= self.depth:
            return True
        return False
    def nextValue(self, gameState, agentIndex, curDepth, nextAgent, legalAction):
        nextState = gameState.getNextState(agentIndex, legalAction)
        legalValue = float()
        if nextAgent == 0:
            legalValue, _ = self.maxValue(nextState, nextAgent, curDepth + 1)
        else:
            legalValue = self.minValue(nextState, nextAgent, curDepth)
        return legalValue
    def maxValue(self, gameState, agentIndex, curDepth):
        if( self.terminalState(gameState, agentIndex, curDepth) ):
            return self.evaluationFunction(gameState), None
        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        scores = list()
        legalMoves = gameState.getLegalActions(agentIndex)
        for legalAction in legalMoves:
            legalValue = self.nextValue(gameState, agentIndex, curDepth, nextAgent, legalAction)
            scores.append(legalValue)
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)
        return bestScore, legalMoves[chosenIndex]
    def minValue(self, gameState, agentIndex, curDepth):
        if( self.terminalState(gameState, agentIndex, curDepth) ):
            return self.evaluationFunction(gameState)
        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        scores = list()
        for legalAction in gameState.getLegalActions(agentIndex):
            legalValue = self.nextValue(gameState, agentIndex, curDepth, nextAgent, legalAction)
            scores.append(legalValue)
        meanScore = sum(scores) / len(scores)
        return meanScore
        # End your code (Part 3)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (Part 4).
    """
    # Begin your code (Part 4)
    #raise NotImplementedError("To be implemented")
    pacemanDir = currentGameState.getPacmanState().getDirection()
    pacmanPos = currentGameState.getPacmanPosition()
    ghostsState = [(manhattanDistance(pacmanPos, currentGameState.getGhostPosition(Id)), Id) for Id in range(1, currentGameState.getNumAgents())]
    minGhostDist, minGhostId = (0, 0) if len(ghostsState) == 0 else min(ghostsState)
    ghostsDist = [x for x, y in ghostsState]
    avgGhostDist = 0 if len(ghostsDist) == 0 else sum (ghostsDist) / len(ghostsDist)
    isEat = currentGameState.data.agentStates[0].scaredTimer > 1
    curScore = currentGameState.getScore()
    foodDist = [manhattanDistance(pacmanPos, food) for food in currentGameState.getFood().asList()]
    numFood = currentGameState.getNumFood()
    minFoodDist = 1 if numFood == 0 else min(foodDist)
    avgFoodDist = 1 if numFood == 0 else sum(foodDist) / len(foodDist)
    capsulesDist = [manhattanDistance(pacmanPos, capsule) for capsule in currentGameState.getCapsules()]
    minCapsuleDist = 1 if len(currentGameState.getCapsules()) == 0 else min( capsulesDist )
    wall = currentGameState.hasWall(0,0)
    numAgent = currentGameState.getNumAgents() > 2
    """
    if isEat:
        return -60 * minGhostDist + curScore
    elif minCapsuleDist < 2 and minGhostDist < 3:
        return 300
    elif minGhostDist < 1:
        return curScore - 500
    else:
        return -1 * ( 1 / ( numFood + 1 ) ) * ( minFoodDist +  0.6 * avgFoodDist)  + curScore
    """
    """
    if isEat:
        return -50 * minGhostDist + 100 * curScore
    elif minGhostDist < 2:
        return 0.8 * curScore + avgGhostDist * numAgent
    elif len(currentGameState.getCapsules()) > 0:
        return 1 / minCapsuleDist
    else:
        return 0.5 * (1 / minFoodDist) + 1 / minCapsuleDist + curScore
    """
    #return isEat * -50 * minGhostDist + 100 * curScore + 0.8 * curScore + avgGhostDist * numAgent + 0.5 * (1 / minFoodDist) + 1 / minCapsuleDist + curScore
    """
    if isEat:
        return 1 / minGhostDist
    elif minGhostDist < 2:
        return 1/ (minGhostDist+1)
    elif len(capsulesDist) > 0:
        return 1 / capsulesDist[len(capsulesDist) - 1]
    else:
        return 1 / minFoodDist
    """
    if isEat:
        return -30 * minGhostDist + curScore
    elif minGhostDist < 3:
        return curScore + avgGhostDist * numAgent
    else:
        return -1 * minFoodDist + -20 * minCapsuleDist + curScore
    # End your code (Part 4)

# Abbreviation
better = betterEvaluationFunction
