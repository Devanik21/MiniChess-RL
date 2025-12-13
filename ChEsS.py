import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import random
import pandas as pd
import json
import zipfile
import io
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Tuple, Optional
import ast

# ============================================================================
# Page Config
# ============================================================================
st.set_page_config(
    page_title="♟️ Minichess Arena",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="♟️"
)

st.title("AlphaZero-Inspired Gardner's 5x5 Minichess Arena")
st.markdown("""
Two AI agents battle using AlphaZero-inspired techniques on a 5x5 chessboard.

**Architecture Components:**
- 🌳 **MCTS with UCB** - Monte Carlo Tree Search for strategic planning
- 🧠 **Deep Position Evaluation** - Advanced chess heuristics
- 🎯 **Policy & Value Heads** - Dual output system for move selection
- 🔄 **Self-Play Training** - Agents improve through competition
- ⚡ **Minimax with Alpha-Beta** - Tactical depth analysis
- 🎲 **Hybrid Decision Making** - MCTS + minimax combination
""", unsafe_allow_html=True)

# ============================================================================
# Gardner's 5x5 Minichess Environment
# ============================================================================

@dataclass
class Move:
    start: Tuple[int, int]
    end: Tuple[int, int]
    piece: str
    captured: Optional[str] = None
    promotion: Optional[str] = None
    is_check: bool = False
    is_checkmate: bool = False
    
    def __hash__(self):
        return hash((self.start, self.end, self.piece, self.captured, self.promotion))
    
    def __eq__(self, other):
        return (self.start == other.start and self.end == other.end and 
                self.piece == other.piece)
    
    def to_notation(self):
        """Convert to algebraic notation"""
        cols = 'abcde'
        s = f"{cols[self.start[1]]}{5-self.start[0]}"
        e = f"{cols[self.end[1]]}{5-self.end[0]}"
        notation = f"{s}{e}"
        if self.promotion:
            notation += f"={self.promotion.upper()}"
        if self.is_checkmate:
            notation += "#"
        elif self.is_check:
            notation += "+"
        return notation

class Minichess:
    """Gardner's 5x5 Minichess - Full chess rules on 5x5 board"""
    
    # Piece values for evaluation
    PIECE_VALUES = {
        'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 20000,
        'p': -100, 'n': -320, 'b': -330, 'r': -500, 'q': -900, 'k': -20000
    }
    
    def __init__(self):
        self.board_size = 5
        self.reset()
    
    def reset(self):
        """Initialize Gardner's 5x5 Minichess starting position"""
        # Starting position for Gardner's Minichess
        # White (uppercase) at bottom (rows 3-4), Black (lowercase) at top (rows 0-1)
        self.board = np.array([
            ['k', 'q', 'b', 'n', 'r'],  # Row 0 - Black back rank
            ['p', 'p', 'p', 'p', 'p'],  # Row 1 - Black pawns
            ['.', '.', '.', '.', '.'],  # Row 2 - Empty
            ['P', 'P', 'P', 'P', 'P'],  # Row 3 - White pawns
            ['K', 'Q', 'B', 'N', 'R']   # Row 4 - White back rank
        ])
        
        self.current_player = 1  # 1 = White, 2 = Black
        self.game_over = False
        self.winner = None
        self.move_history = []
        self.move_count = 0
        return self.get_state()
    
    def get_state(self):
        """Return hashable board state with native Python types"""
        # Crucial fix: Convert numpy strings to Python strings for safe JSON serialization
        return tuple(tuple(str(c) for c in row) for row in self.board)
    
    def copy(self):
        """Deep copy of game state"""
        new_game = Minichess()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.game_over = self.game_over
        new_game.winner = self.winner
        new_game.move_history = self.move_history.copy()
        new_game.move_count = self.move_count
        return new_game
    
    def is_white_piece(self, piece):
        return piece.isupper() and piece != '.'
    
    def is_black_piece(self, piece):
        return piece.islower() and piece != '.'
    
    def is_enemy(self, piece, player):
        if player == 1:  # White
            return self.is_black_piece(piece)
        else:  # Black
            return self.is_white_piece(piece)
    
    def is_friendly(self, piece, player):
        if player == 1:  # White
            return self.is_white_piece(piece)
        else:  # Black
            return self.is_black_piece(piece)
    
    def get_piece_moves(self, row, col, check_legal=True):
        """Get all pseudo-legal moves for a piece"""
        piece = self.board[row, col]
        if piece == '.' or not self.is_friendly(piece, self.current_player):
            return []
        
        moves = []
        piece_type = piece.upper()
        
        if piece_type == 'P':
            moves = self._get_pawn_moves(row, col)
        elif piece_type == 'N':
            moves = self._get_knight_moves(row, col)
        elif piece_type == 'B':
            moves = self._get_bishop_moves(row, col)
        elif piece_type == 'R':
            moves = self._get_rook_moves(row, col)
        elif piece_type == 'Q':
            moves = self._get_queen_moves(row, col)
        elif piece_type == 'K':
            moves = self._get_king_moves(row, col)
        
        # Filter out moves that leave king in check
        if check_legal:
            legal_moves = []
            for move in moves:
                test_game = self.copy()
                test_game._make_move_internal(move)
                if not test_game._is_in_check(self.current_player):
                    legal_moves.append(move)
            return legal_moves
        
        return moves
    
    def _get_pawn_moves(self, row, col):
        """Get pawn moves (including captures)"""
        moves = []
        piece = self.board[row, col]
        
        if self.current_player == 1:  # White pawns move up (decreasing row)
            direction = -1
            start_row = 3
            promotion_row = 0
        else:  # Black pawns move down (increasing row)
            direction = 1
            start_row = 1
            promotion_row = 4
        
        # Forward move
        new_row = row + direction
        if 0 <= new_row < 5 and self.board[new_row, col] == '.':
            if new_row == promotion_row:
                # Promotion
                for promo in ['Q', 'R', 'B', 'N']:
                    moves.append(Move((row, col), (new_row, col), piece, promotion=promo))
            else:
                moves.append(Move((row, col), (new_row, col), piece))
        
        # Captures
        for dc in [-1, 1]:
            new_col = col + dc
            if 0 <= new_row < 5 and 0 <= new_col < 5:
                target = self.board[new_row, new_col]
                if target != '.' and self.is_enemy(target, self.current_player):
                    if new_row == promotion_row:
                        for promo in ['Q', 'R', 'B', 'N']:
                            moves.append(Move((row, col), (new_row, new_col), piece, 
                                            captured=target, promotion=promo))
                    else:
                        moves.append(Move((row, col), (new_row, new_col), piece, captured=target))
        
        return moves
    
    def _get_knight_moves(self, row, col):
        """Get knight moves"""
        moves = []
        piece = self.board[row, col]
        knight_moves = [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1)
        ]
        
        for dr, dc in knight_moves:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 5 and 0 <= new_col < 5:
                target = self.board[new_row, new_col]
                if target == '.' or self.is_enemy(target, self.current_player):
                    captured = target if target != '.' else None
                    moves.append(Move((row, col), (new_row, new_col), piece, captured=captured))
        
        return moves
    
    def _get_sliding_moves(self, row, col, directions):
        """Helper for sliding pieces (bishop, rook, queen)"""
        moves = []
        piece = self.board[row, col]
        
        for dr, dc in directions:
            for i in range(1, 5):
                new_row, new_col = row + dr * i, col + dc * i
                if not (0 <= new_row < 5 and 0 <= new_col < 5):
                    break
                
                target = self.board[new_row, new_col]
                if target == '.':
                    moves.append(Move((row, col), (new_row, new_col), piece))
                elif self.is_enemy(target, self.current_player):
                    moves.append(Move((row, col), (new_row, new_col), piece, captured=target))
                    break
                else:
                    break
        
        return moves
    
    def _get_bishop_moves(self, row, col):
        """Get bishop moves"""
        return self._get_sliding_moves(row, col, [(-1, -1), (-1, 1), (1, -1), (1, 1)])
    
    def _get_rook_moves(self, row, col):
        """Get rook moves"""
        return self._get_sliding_moves(row, col, [(-1, 0), (1, 0), (0, -1), (0, 1)])
    
    def _get_queen_moves(self, row, col):
        """Get queen moves (combination of rook and bishop)"""
        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1), (-1, 0), (1, 0), (0, -1), (0, 1)]
        return self._get_sliding_moves(row, col, directions)
    
    def _get_king_moves(self, row, col):
        """Get king moves"""
        moves = []
        piece = self.board[row, col]
        
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                new_row, new_col = row + dr, col + dc
                if 0 <= new_row < 5 and 0 <= new_col < 5:
                    target = self.board[new_row, new_col]
                    if target == '.' or self.is_enemy(target, self.current_player):
                        captured = target if target != '.' else None
                        moves.append(Move((row, col), (new_row, new_col), piece, captured=captured))
        
        return moves
    
    def get_all_valid_moves(self):
        """Get all legal moves for current player"""
        all_moves = []
        for row in range(5):
            for col in range(5):
                piece = self.board[row, col]
                if piece != '.' and self.is_friendly(piece, self.current_player):
                    moves = self.get_piece_moves(row, col)
                    all_moves.extend(moves)
        return all_moves
    
    def _find_king(self, player):
        """Find the king position for a player"""
        king = 'K' if player == 1 else 'k'
        for row in range(5):
            for col in range(5):
                if self.board[row, col] == king:
                    return (row, col)
        return None
    
    def _is_square_attacked(self, row, col, by_player):
        """Check if a square is attacked by a player"""
        # Temporarily switch player to check their attacks
        original_player = self.current_player
        self.current_player = by_player
        
        for r in range(5):
            for c in range(5):
                piece = self.board[r, c]
                if piece != '.' and self.is_friendly(piece, by_player):
                    # Get pseudo-legal moves without checking legality
                    moves = self.get_piece_moves(r, c, check_legal=False)
                    for move in moves:
                        if move.end == (row, col):
                            self.current_player = original_player
                            return True
        
        self.current_player = original_player
        return False
    
    def _is_in_check(self, player):
        """Check if player's king is in check"""
        king_pos = self._find_king(player)
        if not king_pos:
            return False
        opponent = 3 - player
        return self._is_square_attacked(king_pos[0], king_pos[1], opponent)
    
    def _make_move_internal(self, move):
        """Make a move without checking legality (for simulation)"""
        sr, sc = move.start
        er, ec = move.end
        
        # Handle promotion
        if move.promotion:
            piece = move.promotion if self.current_player == 1 else move.promotion.lower()
        else:
            piece = self.board[sr, sc]
        
        self.board[er, ec] = piece
        self.board[sr, sc] = '.'
    
    def make_move(self, move: Move):
        """Execute a move and return (next_state, reward, done)"""
        if self.game_over:
            return self.get_state(), 0, True
        
        sr, sc = move.start
        er, ec = move.end
        
        # Calculate reward
        reward = 0
        if move.captured:
            reward = abs(self.PIECE_VALUES.get(move.captured, 0)) / 100
        if move.promotion:
            reward += 5
        
        # Make the move
        self._make_move_internal(move)
        self.move_history.append(move)
        self.move_count += 1
        
        # Switch player
        self.current_player = 3 - self.current_player
        
        # Check for check/checkmate/stalemate
        opponent_moves = self.get_all_valid_moves()
        is_check = self._is_in_check(self.current_player)
        
        if not opponent_moves:
            self.game_over = True
            if is_check:
                # Checkmate
                self.winner = 3 - self.current_player
                reward = 100
                move.is_checkmate = True
            else:
                # Stalemate
                self.winner = 0
                reward = 0
        elif is_check:
            move.is_check = True
        
        # Draw by insufficient material or move limit
        if self.move_count >= 100:
            self.game_over = True
            self.winner = 0
        
        return self.get_state(), reward, self.game_over
    
    def evaluate_position(self, player):
        """Evaluate position from player's perspective"""
        if self.winner == player:
            return 100000
        if self.winner == (3 - player):
            return -100000
        if self.winner == 0:
            return 0
        
        score = 0
        
        # Piece-Square Tables for Minichess (5x5)
        pawn_table = [
            [0,  0,  0,  0,  0],
            [50, 50, 50, 50, 50],
            [10, 10, 20, 10, 10],
            [5,  5, 10,  5,  5],
            [0,  0,  0,  0,  0]
        ]
        
        knight_table = [
            [-50, -40, -30, -40, -50],
            [-40, -20,  0,  -20, -40],
            [-30,  0,  10,  0, -30],
            [-40, -20,  0, -20, -40],
            [-50, -40, -30, -40, -50]
        ]
        
        center_table = [
            [0, 0, 0, 0, 0],
            [0, 5, 5, 5, 0],
            [0, 5, 10, 5, 0],
            [0, 5, 5, 5, 0],
            [0, 0, 0, 0, 0]
        ]
        
        for row in range(5):
            for col in range(5):
                piece = self.board[row, col]
                if piece == '.':
                    continue
                
                is_mine = self.is_friendly(piece, player)
                multiplier = 1 if is_mine else -1
                
                # Material value
                piece_value = abs(self.PIECE_VALUES.get(piece, 0))
                score += multiplier * piece_value
                
                # Positional bonuses
                piece_type = piece.upper()
                if piece_type == 'P':
                    # Adjust row index based on piece color
                    if piece.isupper():  # White
                        pos_bonus = pawn_table[row][col]
                    else:  # Black
                        pos_bonus = pawn_table[4-row][col]
                    score += multiplier * pos_bonus
                elif piece_type == 'N':
                    score += multiplier * knight_table[row][col]
                elif piece_type in ['B', 'R', 'Q']:
                    score += multiplier * center_table[row][col] * 2
                elif piece_type == 'K':
                    # King safety (prefer edges in endgame, center in opening)
                    score += multiplier * center_table[row][col]
        
        # Mobility bonus
        self.current_player = player
        my_moves = len(self.get_all_valid_moves())
        self.current_player = 3 - player
        opp_moves = len(self.get_all_valid_moves())
        self.current_player = player
        
        score += (my_moves - opp_moves) * 10
        
        # Check bonus
        if self._is_in_check(3 - player):
            score += 50
        
        return score

# ============================================================================
# MCTS Node (AlphaZero Core Component)
# ============================================================================

class MCTSNode:
    def __init__(self, game_state, parent=None, move=None, prior=1.0):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.prior = prior
        
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
    
    def value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0
    
    def ucb_score(self, parent_visits, c_puct=1.5):
        if self.visit_count == 0:
            q_value = 0
        else:
            q_value = self.value()
        
        u_value = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return q_value + u_value
    
    def select_child(self, c_puct=1.5):
        return max(self.children.values(), 
                   key=lambda child: child.ucb_score(self.visit_count, c_puct))
    
    def expand(self, game, policy_priors):
        valid_moves = game.get_all_valid_moves()
        if not valid_moves:
            return
        
        total_prior = sum(policy_priors.values())
        if total_prior == 0:
            total_prior = len(valid_moves)
        
        for move in valid_moves:
            prior = policy_priors.get(move, 1.0) / total_prior
            child_game = game.copy()
            child_game.make_move(move)
            self.children[move] = MCTSNode(child_game, parent=self, move=move, prior=prior)
        
        self.is_expanded = True
    
    def backup(self, value):
        self.visit_count += 1
        self.value_sum += value
        if self.parent:
            self.parent.backup(-value)

# ============================================================================
# AlphaZero-Inspired Agent
# ============================================================================

class Agent:
    def __init__(self, player_id, lr=0.3, gamma=0.99, epsilon=1.0):
        self.player_id = player_id
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.96
        self.epsilon_min = 0.01
        
        self.mcts_simulations = 100
        self.c_puct = 1.4
        self.minimax_depth = 3
        
        self.policy_table = defaultdict(lambda: defaultdict(float))
        
        self.wins = 0
        self.losses = 0
        self.draws = 0
        
        self.game_history = []
    
    def get_policy_priors(self, game):
        state = game.get_state()
        moves = game.get_all_valid_moves()
        priors = {}
        
        for move in moves:
            if state in self.policy_table and move in self.policy_table[state]:
                priors[move] = self.policy_table[state][move]
            else:
                prior = 1.0
                if move.captured:
                    piece_val = abs(Minichess.PIECE_VALUES.get(move.captured, 0))
                    prior += piece_val / 100
                if move.promotion:
                    prior += 3.0
                if move.is_check:
                    prior += 2.0
                priors[move] = prior
        
        return priors
    
    def mcts_search(self, game, num_simulations):
        root = MCTSNode(game.copy())
        
        for _ in range(num_simulations):
            node = root
            search_game = game.copy()
            search_path = [node]
            
            while node.is_expanded and node.children:
                node = node.select_child(self.c_puct)
                search_game.make_move(node.move)
                search_path.append(node)
            
            if not search_game.game_over:
                policy_priors = self.get_policy_priors(search_game)
                node.expand(search_game, policy_priors)
            
            value = self._evaluate_leaf(search_game)
            node.backup(value)
        
        return root
    
    def _evaluate_leaf(self, game):
        if game.game_over:
            if game.winner == self.player_id:
                return 1.0
            elif game.winner == (3 - self.player_id):
                return -1.0
            return 0.0
        
        score = self._minimax(game, self.minimax_depth, -float('inf'), float('inf'), True)
        return np.tanh(score / 1000)
    
    def _minimax(self, game, depth, alpha, beta, maximizing):
        if depth == 0 or game.game_over:
            return game.evaluate_position(self.player_id)
        
        moves = game.get_all_valid_moves()
        if not moves:
            return game.evaluate_position(self.player_id)
        
        # Limit branching
        search_candidates = moves[:8]
        
        if maximizing:
            max_eval = -float('inf')
            for move in search_candidates:
                sim_game = game.copy()
                sim_game.make_move(move)
                eval_score = self._minimax(sim_game, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in search_candidates:
                sim_game = game.copy()
                sim_game.make_move(move)
                eval_score = self._minimax(sim_game, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval
    
    def choose_action(self, game, training=True):
        moves = game.get_all_valid_moves()
        if not moves:
            return None
        
        if training and random.random() < self.epsilon:
            return random.choice(moves)
        
        root = self.mcts_search(game, self.mcts_simulations)
        
        if not root.children:
            return random.choice(moves)
        
        best_move = max(root.children.items(), key=lambda x: x[1].visit_count)[0]
        
        state = game.get_state()
        total_visits = sum(child.visit_count for child in root.children.values())
        for move, child in root.children.items():
            self.policy_table[state][move] = child.visit_count / total_visits
        
        return best_move
    
    def update_from_game(self, game_data, result):
        for state, move, player in game_data:
            if player != self.player_id:
                continue
            
            if result == self.player_id:
                reward = 1.0
            elif result == 0:
                reward = 0.0
            else:
                reward = -1.0
            
            current_policy = self.policy_table[state][move]
            self.policy_table[state][move] = current_policy + self.lr * (reward - current_policy)
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def reset_stats(self):
        self.wins = 0
        self.losses = 0
        self.draws = 0

# ============================================================================
# Training System
# ============================================================================

def play_game(env, agent1, agent2, training=True):
    env.reset()
    game_history = []
    agents = {1: agent1, 2: agent2}
    
    move_count = 0
    max_moves = 100
    
    while not env.game_over and move_count < max_moves:
        current_player = env.current_player
        agent = agents[current_player]
        
        state = env.get_state()
        move = agent.choose_action(env, training)
        
        if move is None:
            break
        
        game_history.append((state, move, current_player))
        env.make_move(move)
        move_count += 1
    
    if env.winner == 1:
        agent1.wins += 1
        agent2.losses += 1
        if training:
            agent1.update_from_game(game_history, 1)
            agent2.update_from_game(game_history, 1)
    elif env.winner == 2:
        agent2.wins += 1
        agent1.losses += 1
        if training:
            agent1.update_from_game(game_history, 2)
            agent2.update_from_game(game_history, 2)
    else:
        agent1.draws += 1
        agent2.draws += 1
        if training:
            agent1.update_from_game(game_history, 0)
            agent2.update_from_game(game_history, 0)
    
    return env.winner

# ============================================================================
# Visualization
# ============================================================================

def visualize_board(board, title="Minichess Board", last_move=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Unicode chess pieces
    piece_symbols = {
        'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
        'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'
    }
    
    # Draw board
    for row in range(5):
        for col in range(5):
            color = '#F0D9B5' if (row + col) % 2 == 0 else '#B58863'
            
            # Highlight last move
            if last_move and ((row, col) == last_move.start or (row, col) == last_move.end):
                color = '#BACA44'
            
            square = plt.Rectangle((col, 4-row), 1, 1, facecolor=color)
            ax.add_patch(square)
            
            piece = board[row, col]
            if piece != '.':
                symbol = piece_symbols.get(piece, piece)
                color = '#FFFFFF' if piece.isupper() else '#000000'
                ax.text(col + 0.5, 4-row + 0.5, symbol, 
                       ha='center', va='center', fontsize=36, color=color)
    
    # Add coordinates
    for i in range(5):
        ax.text(-0.3, 4-i+0.5, str(i+1), ha='center', va='center', fontsize=12)
        ax.text(i+0.5, -0.3, 'abcde'[i], ha='center', va='center', fontsize=12)
    
    ax.set_xlim(-0.5, 5)
    ax.set_ylim(-0.5, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    return fig

# ============================================================================
# Serialization
# ============================================================================

# ============================================================================
# Serialization (Fixed)
# ============================================================================

def serialize_move(move):
    return {
        "s": [int(x) for x in move.start],
        "e": [int(x) for x in move.end],
        "p": str(move.piece),
        "c": str(move.captured) if move.captured else None,
        "pr": str(move.promotion) if move.promotion else None
    }

def deserialize_move(data):
    return Move(
        start=tuple(data["s"]),
        end=tuple(data["e"]),
        piece=data["p"],
        captured=data.get("c"),
        promotion=data.get("pr")
    )

def create_agents_zip(agent1, agent2, config):
    def serialize_agent(agent, role_name):
        clean_policy = {}
        # Make a copy to avoid runtime modification errors
        current_policies = agent.policy_table.copy()
        
        for state, moves in current_policies.items():
            try:
                # Ensure state is stringified safely
                state_str = str(state)
                clean_policy[state_str] = {}
                
                for move, value in moves.items():
                    # Serialize move object to JSON string to use as dict key
                    move_json_str = json.dumps(serialize_move(move))
                    clean_policy[state_str][move_json_str] = float(value)
            except Exception:
                continue
        
        return {
            "metadata": {"role": role_name, "version": "2.1"},
            "policy_table": clean_policy,
            "epsilon": float(agent.epsilon),
            "wins": int(agent.wins),
            "losses": int(agent.losses),
            "draws": int(agent.draws),
            "mcts_sims": int(agent.mcts_simulations)
        }
    
    data1 = serialize_agent(agent1, "White")
    data2 = serialize_agent(agent2, "Black")
    
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("agent1.json", json.dumps(data1, indent=2))
        zf.writestr("agent2.json", json.dumps(data2, indent=2))
        zf.writestr("config.json", json.dumps(config, indent=2))
    
    buffer.seek(0)
    return buffer

def load_agents_from_zip(uploaded_file):
    try:
        with zipfile.ZipFile(uploaded_file, "r") as zf:
            files = zf.namelist()
            if not all(f in files for f in ["agent1.json", "agent2.json", "config.json"]):
                st.error("❌ Corrupt File: Missing files in zip.")
                return None, None, None, 0
            
            a1_data = json.loads(zf.read("agent1.json").decode('utf-8'))
            a2_data = json.loads(zf.read("agent2.json").decode('utf-8'))
            config = json.loads(zf.read("config.json").decode('utf-8'))
            
            def restore_agent(agent, data):
                agent.epsilon = data.get('epsilon', 0.1)
                agent.wins = data.get('wins', 0)
                agent.losses = data.get('losses', 0)
                agent.draws = data.get('draws', 0)
                agent.mcts_simulations = data.get('mcts_sims', 50)
                
                # Reset policy table
                agent.policy_table = defaultdict(lambda: defaultdict(float))
                loaded_policies_count = 0
                
                policy_data = data.get('policy_table', {})
                
                for state_str, moves_dict in policy_data.items():
                    try:
                        # Convert string "((...))" back to tuple
                        state = ast.literal_eval(state_str)
                        
                        for move_json_str, value in moves_dict.items():
                            # Convert JSON string key back to dict, then to Move object
                            move_dict = json.loads(move_json_str)
                            move = deserialize_move(move_dict)
                            agent.policy_table[state][move] = value
                        
                        loaded_policies_count += 1
                    except Exception:
                        # Skip malformed entries but continue loading others
                        continue
                        
                return loaded_policies_count
            
            # Recreate Agent 1
            agent1 = Agent(1, config.get('lr1', 0.3), config.get('gamma1', 0.95))
            count1 = restore_agent(agent1, a1_data)
            
            # Recreate Agent 2
            agent2 = Agent(2, config.get('lr2', 0.3), config.get('gamma2', 0.95))
            count2 = restore_agent(agent2, a2_data)
            
            return agent1, agent2, config, count1 + count2
            
    except Exception as e:
        st.error(f"❌ Error loading brain: {str(e)}")
        return None, None, None, 0


# ============================================================================
# Streamlit UI
# ============================================================================

st.sidebar.header("⚙️ Configuration")

with st.sidebar.expander("1. Agent 1 (White) Parameters", expanded=True):
    lr1 = st.slider("Learning Rate α₁", 0.1, 1.0, 0.3, 0.05)
    gamma1 = st.slider("Discount Factor γ₁", 0.8, 0.99, 0.95, 0.01)
    mcts_sims1 = st.slider("MCTS Simulations₁", 10, 500, 100, 10)
    minimax_depth1 = st.slider("Minimax Depth₁", 1, 6, 3, 1)

with st.sidebar.expander("2. Agent 2 (Black) Parameters", expanded=True):
    lr2 = st.slider("Learning Rate α₂", 0.1, 1.0, 0.3, 0.05)
    gamma2 = st.slider("Discount Factor γ₂", 0.8, 0.99, 0.95, 0.01)
    mcts_sims2 = st.slider("MCTS Simulations₂", 10, 500, 80, 10)
    minimax_depth2 = st.slider("Minimax Depth₂", 1, 6, 2, 1)

with st.sidebar.expander("3. Training Configuration", expanded=True):
    episodes = st.number_input("Training Episodes", 10, 5000, 500, 10)
    update_freq = st.number_input("Update Every N Games", 1, 200, 25, 1)

with st.sidebar.expander("4. Brain Storage 💾", expanded=False):
    if 'agent1' in st.session_state and st.session_state.agent1:
        st.markdown("### 🧠 Neural Synchronization")
        st.caption("Balance agents by copying the smarter brain.")
        col_sync1, col_sync2 = st.columns(2)
        
        if col_sync1.button("White → Black", help="Copy Agent 1's brain to Agent 2"):
            st.session_state.agent2.policy_table = deepcopy(st.session_state.agent1.policy_table)
            st.session_state.agent2.epsilon = st.session_state.agent1.epsilon
            st.toast("Agent 2 (Black) synchronized!", icon="⚫")
        
        if col_sync2.button("Black → White", help="Copy Agent 2's brain to Agent 1"):
            st.session_state.agent1.policy_table = deepcopy(st.session_state.agent2.policy_table)
            st.session_state.agent1.epsilon = st.session_state.agent2.epsilon
            st.toast("Agent 1 (White) synchronized!", icon="⚪")
        
        st.markdown("---")
        
        config = {
            "lr1": lr1, "gamma1": gamma1, "mcts_sims1": mcts_sims1, "minimax_depth1": minimax_depth1,
            "lr2": lr2, "gamma2": gamma2, "mcts_sims2": mcts_sims2, "minimax_depth2": minimax_depth2,
            "training_history": st.session_state.get('training_history', None)
        }
        
        zip_buffer = create_agents_zip(st.session_state.agent1, st.session_state.agent2, config)
        st.download_button(
            label="💾 Download AlphaZero Agents",
            data=zip_buffer,
            file_name="minichess_alphazero.zip",
            mime="application/zip",
            use_container_width=True
        )
    else:
        st.info("Train agents first to enable saving")
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader("📤 Upload Saved Agents (.zip)", type="zip")
    if uploaded_file is not None:
        if st.button("🔄 Load Agents", use_container_width=True):
            with st.spinner("Restoring neural pathways..."):
                a1, a2, cfg, count = load_agents_from_zip(uploaded_file)
                if a1 and a2:
                    st.session_state.agent1 = a1
                    st.session_state.agent2 = a2
                    st.session_state.training_history = cfg.get("training_history", None)
                    
                    # Force update of the environment agents
                    st.session_state.env = Minichess()
                    
                    # Success message
                    st.toast(f"✅ Brain Loaded! {count} memories restored.", icon="🧠")
                    st.success(f"Successfully loaded {count} policies! You can now Play or Watch.")
                    
                    # Wait a moment then rerun to unlock the UI
                    import time
                    time.sleep(1.5)
                    st.rerun()

train_button = st.sidebar.button("🚀 Begin Self-Play Training", 
                                 use_container_width=True, type="primary")

if st.sidebar.button("🧹 Reset Arena", use_container_width=True):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.cache_data.clear()
    st.rerun()

# Initialize
if 'env' not in st.session_state:
    st.session_state.env = Minichess()

if 'agent1' not in st.session_state:
    st.session_state.agent1 = Agent(1, lr1, gamma1)
    st.session_state.agent1.mcts_simulations = mcts_sims1
    st.session_state.agent1.minimax_depth = minimax_depth1
    
    st.session_state.agent2 = Agent(2, lr2, gamma2)
    st.session_state.agent2.mcts_simulations = mcts_sims2
    st.session_state.agent2.minimax_depth = minimax_depth2

agent1 = st.session_state.agent1
agent2 = st.session_state.agent2
env = st.session_state.env

# Update parameters
agent1.mcts_simulations = mcts_sims1
agent1.minimax_depth = minimax_depth1
agent2.mcts_simulations = mcts_sims2
agent2.minimax_depth = minimax_depth2

# Display stats
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("⚪ Agent 1 (White)", 
             f"Policies: {len(agent1.policy_table)}", 
             f"ε={agent1.epsilon:.4f}")
    st.metric("Wins", agent1.wins)
    st.caption(f"MCTS: {agent1.mcts_simulations} | Depth: {agent1.minimax_depth}")

with col2:
    st.metric("⚫ Agent 2 (Black)", 
             f"Policies: {len(agent2.policy_table)}", 
             f"ε={agent2.epsilon:.4f}")
    st.metric("Wins", agent2.wins)
    st.caption(f"MCTS: {agent2.mcts_simulations} | Depth: {agent2.minimax_depth}")

with col3:
    total = agent1.wins + agent2.wins + agent1.draws
    st.metric("Total Games", total)
    st.metric("Draws", agent1.draws)

st.markdown("---")

# Training
if train_button:
    st.subheader("🎯 Self-Play Training")
    
    status = st.empty()
    progress_bar = st.progress(0)
    
    agent1.reset_stats()
    agent2.reset_stats()
    
    history = {
        'agent1_wins': [], 'agent2_wins': [], 'draws': [],
        'agent1_epsilon': [], 'agent2_epsilon': [],
        'agent1_policies': [], 'agent2_policies': [],
        'episode': []
    }
    
    for ep in range(1, episodes + 1):
        winner = play_game(env, agent1, agent2, training=True)
        
        agent1.decay_epsilon()
        agent2.decay_epsilon()
        
        if ep % update_freq == 0:
            history['agent1_wins'].append(agent1.wins)
            history['agent2_wins'].append(agent2.wins)
            history['draws'].append(agent1.draws)
            history['agent1_epsilon'].append(agent1.epsilon)
            history['agent2_epsilon'].append(agent2.epsilon)
            history['agent1_policies'].append(len(agent1.policy_table))
            history['agent2_policies'].append(len(agent2.policy_table))
            history['episode'].append(ep)
            
            progress = ep / episodes
            progress_bar.progress(progress)
            
            status.markdown(f"""
            | Metric | Agent 1 (White) | Agent 2 (Black) |
            |:-------|:-------------:|:---------------:|
            | **Wins** | {agent1.wins} | {agent2.wins} |
            | **Epsilon** | {agent1.epsilon:.4f} | {agent2.epsilon:.4f} |
            | **Policies** | {len(agent1.policy_table):,} | {len(agent2.policy_table):,} |
            
            **Game {ep}/{episodes}** ({progress*100:.1f}%) | **Draws:** {agent1.draws}
            """)
    
    progress_bar.progress(1.0)
    st.toast("Training Complete! 🎉", icon="✨")
    st.session_state.training_history = history
    
    import time
    with st.spinner("Saving brain state..."):
        time.sleep(1)
        st.rerun()

# Training charts
if 'training_history' in st.session_state and st.session_state.training_history:
    history = st.session_state.training_history
    
    if isinstance(history, dict) and 'episode' in history and len(history['episode']) > 0:
        st.subheader("📊 Training Analytics")
        df = pd.DataFrame(history)
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.write("#### Win/Draw Distribution")
            cols_needed = ['episode', 'agent1_wins', 'agent2_wins', 'draws']
            if all(col in df.columns for col in cols_needed):
                chart_data = df[cols_needed].set_index('episode')
                st.line_chart(chart_data)
        
        with chart_col2:
            st.write("#### Exploration Rate (Epsilon)")
            cols_needed = ['episode', 'agent1_epsilon', 'agent2_epsilon']
            if all(col in df.columns for col in cols_needed):
                chart_data = df[cols_needed].set_index('episode')
                st.line_chart(chart_data)
        
        st.write("#### Policy Network Growth")
        cols_needed = ['episode', 'agent1_policies', 'agent2_policies']
        if all(col in df.columns for col in cols_needed):
            chart_data = df[cols_needed].set_index('episode')
            st.line_chart(chart_data)

# Demo Game
if 'agent1' in st.session_state and len(agent1.policy_table) > 50:
    st.markdown("---")
    st.subheader("⚔️ Championship Match")
    st.info("Watch the trained AlphaZero agents compete!")
    
    if st.button("🎬 Watch Battle!", use_container_width=True):
        sim_env = Minichess()
        board_placeholder = st.empty()
        move_text = st.empty()
        
        agents = {1: agent1, 2: agent2}
        move_num = 0
        
        with st.spinner("AlphaZero agents thinking..."):
            while not sim_env.game_over and move_num < 100:
                current_player = sim_env.current_player
                move = agents[current_player].choose_action(sim_env, training=False)
                
                if move is None:
                    break
                
                last_move = move
                sim_env.make_move(move)
                move_num += 1
                
                player_name = "White" if current_player == 1 else "Black"
                move_text.caption(f"Move {move_num}: {player_name} plays {move.to_notation()}")
                
                fig = visualize_board(sim_env.board, 
                                     f"{player_name}'s Move #{move_num}", last_move)
                board_placeholder.pyplot(fig)
                plt.close(fig)
                
                import time
                time.sleep(1)
        
        if sim_env.winner == 1:
            st.success("🏆 Agent 1 (White) Wins!")
        elif sim_env.winner == 2:
            st.error("🏆 Agent 2 (Black) Wins!")
        else:
            st.warning("🤝 Draw!")

# ============================================================================
# Human vs AI Arena
# ============================================================================

st.markdown("---")
st.header("🎮 Challenge AlphaZero")

if len(agent1.policy_table) > 50:
    col_h1, col_h2, col_h3 = st.columns([1, 1, 1])
    with col_h1:
        opponent = st.selectbox("Your Opponent", ["Agent 1 (White)", "Agent 2 (Black)"])
    with col_h2:
        color_choice = st.selectbox("Your Color", ["White", "Black"])
    with col_h3:
        st.write("")
        if st.button("🎯 Start Game", use_container_width=True, type="primary"):
            st.session_state.human_env = Minichess()
            st.session_state.human_game_active = True
            
            if "Agent 1" in opponent:
                st.session_state.ai_agent = agent1
                st.session_state.ai_player_id = 1 if color_choice == "Black" else 2
            else:
                st.session_state.ai_agent = agent2
                st.session_state.ai_player_id = 2 if color_choice == "White" else 1
            
            st.session_state.human_player_id = 3 - st.session_state.ai_player_id
            st.session_state.selected_piece = None
            st.rerun()
    
    if 'human_env' in st.session_state and st.session_state.human_game_active:
        h_env = st.session_state.human_env
        
        # AI turn
        if h_env.current_player == st.session_state.ai_player_id and not h_env.game_over:
            with st.spinner("🤖 AlphaZero calculating..."):
                import time
                time.sleep(0.5)
                ai_move = st.session_state.ai_agent.choose_action(h_env, training=False)
                if ai_move:
                    h_env.make_move(ai_move)
                    st.rerun()
        
        # Status
        if h_env.game_over:
            if h_env.winner == st.session_state.human_player_id:
                st.success("🎉 YOU WIN! You defeated AlphaZero!")
            elif h_env.winner == st.session_state.ai_player_id:
                st.error("😮 AlphaZero Wins!")
            else:
                st.warning("🤝 Draw!")
        else:
            turn = "Your Turn" if h_env.current_player == st.session_state.human_player_id else "AI Thinking..."
            player_color = "White" if h_env.current_player == 1 else "Black"
            st.caption(f"**{turn}** ({player_color} to move)")
        
        # Display board
        last_move = h_env.move_history[-1] if h_env.move_history else None
        fig = visualize_board(h_env.board, "Human vs AlphaZero", last_move)
        st.pyplot(fig)
        plt.close(fig)
        
        # Move selection for human
        if (not h_env.game_over and 
            h_env.current_player == st.session_state.human_player_id):
            
            st.write("**Select your piece to move:**")
            valid_moves = h_env.get_all_valid_moves()
            
            # Get unique starting positions
            start_positions = list(set([m.start for m in valid_moves]))
            
            # Show pieces you can move
            cols = st.columns(min(len(start_positions), 5))
            for idx, pos in enumerate(start_positions):
                piece = h_env.board[pos[0], pos[1]]
                piece_symbols = {
                    'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
                    'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'
                }
                symbol = piece_symbols.get(piece, piece)
                coord = f"{'abcde'[pos[1]]}{5-pos[0]}"
                if cols[idx % len(cols)].button(f"{symbol} at {coord}", key=f"select_{pos}"):
                    st.session_state.selected_piece = pos
                    st.rerun()
            
            # Show moves for selected piece
            if 'selected_piece' in st.session_state and st.session_state.selected_piece:
                piece_moves = [m for m in valid_moves if m.start == st.session_state.selected_piece]
                st.write(f"**Available moves:**")
                
                move_cols = st.columns(min(len(piece_moves), 5))
                for idx, move in enumerate(piece_moves):
                    move_desc = move.to_notation()
                    if move.captured:
                        move_desc += " (capture)"
                    if move_cols[idx % len(move_cols)].button(move_desc, key=f"move_{idx}"):
                        h_env.make_move(move)
                        st.session_state.selected_piece = None
                        st.rerun()
else:
    st.info("🎓 Train agents first to unlock Human vs AI mode!")

# Footer
st.markdown("---")
st.caption("Built with AlphaZero-inspired MCTS + Deep Evaluation | Gardner's 5x5 Minichess")
