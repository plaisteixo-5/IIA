######################################################
#Universidade de Brasilia                            #
#Instituto de Ciencias Exatas                        #
#Departamento de Ciencia da Computacao               #
#                                                    #
#Introdução à Inteligência Artificial                #
#Semestre: 2021/1                                    #
#                                                    #
#Aluno : Felipe Fontenele Dos Santos                 #
#Matricula : 19/0027622                              #
#Turma : A                                           #
#Descricao : Projeto realizado na discipĺina de IIA  #
# com o objetivo de compreeender melhor a            #
# implementação de algoritimos de busca na tomada de #
# decisões de uma IA.						         #           
######################################################

# search.py
# ---------
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
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).
    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state
        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state
        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take
        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.
    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    
    stack = util.Stack()
    start_location = problem.getStartState()
    start_node = (start_location, [])
    stack.push(start_node)
    visited_location = [()]

    while not stack.isEmpty():

        position, moves = stack.pop()

        visited_location.append(position)
        if problem.isGoalState(position):
            return moves
        successors = problem.getSuccessors(position)
        
        for successor in successors:
            if successor[0] not in visited_location:
                stack.push((successor[0], moves + [successor[1]]))

    return []

def breadthFirstSearch(problem):

    queue = util.Queue()
    start_location = problem.getStartState()
    start_node = (start_location, [])
    # Structure: ((x, x), ['west'])
    queue.push(start_node)
    # Structure: [((x, x), ['west']), ((y, y), ['south'])]
    visited_location = [()]
    visited_location.append(start_location)

    while not queue.isEmpty():
        position, moves = queue.pop()

        if problem.isGoalState(position):
            return moves

        for successor in problem.getSuccessors(position) :
            if successor[0] in visited_location:
                continue
            visited_location.append(successor[0])
            queue.push((successor[0], moves + [successor[1]])) 
        
    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    priority_queue = util.PriorityQueue()
    visited_location = ([])

    start_location = problem.getStartState()
    priority_queue.push((start_location, [], 0), 0)

    while not priority_queue.isEmpty():
        position, moves, cost_path = priority_queue.pop()

        if position not in visited_location:
            visited_location.append(position)

            if problem.isGoalState(position):
                return moves
            
            for next_pos, movement, cost in problem.getSuccessors(position):
                total_cost = cost + cost_path
                priority_queue.push((next_pos, moves + [movement], total_cost), total_cost)
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    priority_queue = util.PriorityQueue()
    visited_positions = []
    start_location = problem.getStartState()

    priority_queue.push((start_location, [], 0), 0)

    while not priority_queue.isEmpty():
        position, moves, cost_path = priority_queue.pop()

        if position not in visited_positions:
            visited_positions.append(position)

            if problem.isGoalState(position):
                return moves
            
            for next_pos, move, cost in problem.getSuccessors(position):
                total_cost = cost + cost_path
                heuristic_value = total_cost + heuristic(next_pos, problem)
                priority_queue.push((next_pos, moves + [move], total_cost), heuristic_value)

    util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch