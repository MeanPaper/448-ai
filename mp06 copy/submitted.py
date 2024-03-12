import math
import chess.lib
from chess.lib.utils import encode, decode
from chess.lib.heuristics import evaluate
from chess.lib.core import makeMove

###########################################################################################
# Utility function: Determine all the legal moves available for the side.
# This is modified from chess.lib.core.legalMoves:
#  each move has a third element specifying whether the move ends in pawn promotion
def generateMoves(side, board, flags):
    for piece in board[side]:
        fro = piece[:2]
        for to in chess.lib.availableMoves(side, board, piece, flags):
            promote = chess.lib.getPromote(None, side, board, fro, to, single=True)
            yield [fro, to, promote]
            
###########################################################################################
# Example of a move-generating function:
# Randomly choose a move.
def random(side, board, flags, chooser):
    '''
    Return a random move, resulting board, and value of the resulting board.
    Return: (value, moveList, boardList)
      value (int or float): value of the board after making the chosen move
      moveList (list): list with one element, the chosen move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    moves = [ move for move in generateMoves(side, board, flags) ]
    if len(moves) > 0:
        move = chooser(moves)
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        value = evaluate(newboard)
        return (value, [ move ], { encode(*move): {} })
    else:
        return (evaluate(board), [], {})

###########################################################################################
# Stuff you need to write:
# Move-generating functions using minimax, alphabeta, and stochastic search.
def minimax(side, board, flags, depth):
    '''
    Return a minimax-optimal move sequence, tree of all boards evaluated, and value of best path.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''

    next_moves = [n_move for n_move in generateMoves(side, board, flags)] # generate all move for the current flag

    if len(next_moves) <= 0 or depth == 0:
       return(evaluate(board), [], {}) 
    
    final_value = math.inf if side else -math.inf
    final_moveList = []
    final_moveTree = dict()

    # if(side == True): # player 1, minimize, 
    for each_move in next_moves:
      newside, newboard, newflags = makeMove(side, board, each_move[0], each_move[1], flags, each_move[2]) # attempt to make move a direction
      newdepth = depth - 1
      value, movelist, moveTree = minimax(newside, newboard, newflags, newdepth)
      final_moveTree[encode(*each_move)] = moveTree
      if side:
        if(value < final_value): # not gonna lie, this is kinda stupid, if both move are the greatest, why do we just use the one we first seen
          final_value = value
          movelist.insert(0, each_move)
          final_moveList = movelist
      else:
        if(value > final_value):  # not gonna lie, this is kinda stupid, if both move are the greatest, why do we just use the one we first seen
          final_value = value
          movelist.insert(0, each_move) # insert a move to the front, and udpate move
          final_moveList = movelist     

    # else: # player 0, maximize,
      # for each_move in next_moves:
        # newside, newboard, newflags = makeMove(side, board, each_move[0], each_move[1], flags, each_move[2]) # attempt to make move a direction
        # newdepth = depth - 1
        # value, movelist, moveTree = minimax(newside, newboard, newflags, newdepth)
        # final_moveTree[encode(*each_move)] = moveTree


    return (final_value, final_moveList, final_moveTree)

    # raise NotImplementedError("you need to write this!")

def alphabeta(side, board, flags, depth, alpha=-math.inf, beta=math.inf):
    '''
    Return minimax-optimal move sequence, and a tree that exhibits alphabeta pruning.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''
    # raise NotImplementedError("you need to write this!")
    next_moves = [n_move for n_move in generateMoves(side, board, flags)] # generate all move for the current flag
    if len(next_moves) <= 0 or depth == 0:
      return (evaluate(board), [], {})
    
    final_moveList = []
    final_moveTree = dict()
    
    if side: # player 1, min, update beta
      for each_move in next_moves:
        newside, newboard, newflags = makeMove(side, board, each_move[0], each_move[1], flags, each_move[2]) # attempt to make move a direction
        newdepth = depth - 1
        value, moveList, moveTree = alphabeta(newside, newboard, newflags, newdepth, alpha, beta)
        final_moveTree[encode(*each_move)] = moveTree
        if(beta > value): # min(beta, value)
          beta = value  # final_value = updated beta = value
          moveList.insert(0, each_move)
          final_moveList = moveList
        if(beta <= alpha): break
      return (beta, final_moveList, final_moveTree)

    else: # player 0, max update alpha
      for each_move in next_moves:
        newside, newboard, newflags = makeMove(side, board, each_move[0], each_move[1], flags, each_move[2]) # attempt to make move a direction
        newdepth = depth - 1
        value, moveList, moveTree = alphabeta(newside, newboard, newflags, newdepth, alpha, beta)
        final_moveTree[encode(*each_move)] = moveTree
        if(alpha < value):
          alpha = value # final_value = updated beta = value
          moveList.insert(0, each_move)
          final_moveList = moveList
        if(beta <= alpha): break
      return (alpha, final_moveList, final_moveTree)
    
    
    
  
    
    

def stochastic(side, board, flags, depth, breadth, chooser):
    '''
    Choose the best move based on breadth randomly chosen paths per move, of length depth-1.
    Return: (value, moveList, moveTree)
      value (float): average board value of the paths for the best-scoring move
      moveLists (list): any sequence of moves, of length depth, starting with the best move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
      breadth: number of different paths 
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    raise NotImplementedError("you need to write this!")
