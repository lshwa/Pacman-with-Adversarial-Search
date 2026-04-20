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


from game import Directions
import random, util

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
    """

    def getAction(self, gameState: GameState):
        """
        getAction chooses among the best options according to the evaluation function.
        """
        legal_moves = gameState.getLegalActions()

        scores = [self.evaluationFunction(gameState, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [idx for idx in range(len(scores)) if scores[idx] == best_score]
        chosen_index = random.choice(best_indices)

        return legal_moves[chosen_index]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Evaluation function for your reflex agent (question 1).
        """
        successor_game_state = currentGameState.generatePacmanSuccessor(action)
        new_pos = successor_game_state.getPacmanPosition()
        new_food = successor_game_state.getFood()
        new_ghost_states = successor_game_state.getGhostStates()
        new_scared_times = [ghost_state.scaredTimer for ghost_state in new_ghost_states]

        "*** YOUR CODE HERE ***"
        score = 0

        # 이긴 상태면 가장 큰 값 반환
        if successor_game_state.isWin():
            return float('inf')

        # active ghost 중 가장 가까운 거리 확인
        active_ghost_dists = []
        for ghost_state, scared_time in zip(new_ghost_states, new_scared_times):
            if scared_time == 0:
                ghost_dist = util.manhattanDist(new_pos, ghost_state.getPosition())
                active_ghost_dists.append(ghost_dist)

        if active_ghost_dists:
            closest_ghost_dist = min(active_ghost_dists)

            # ghost가 너무 가까우면 큰 페널티
            ghost_penalty = -100000
            if closest_ghost_dist <= 2:
                score += ghost_penalty * (2 - closest_ghost_dist)
            elif closest_ghost_dist == 3:
                score -= 50
            elif closest_ghost_dist == 4:
                score -= 5

        # food를 먹는 행동이면 보상
        food_bonus = 500
        if currentGameState.getNumFood() > successor_game_state.getNumFood():
            score += food_bonus

        # stop은 불리하게 처리
        if action == Directions.STOP:
            score -= 10

        # 가장 가까운 food까지 거리 반영
        min_food_dist = float('inf')
        food_positions = new_food.asList()

        for food_pos in food_positions:
            min_food_dist = min(min_food_dist, util.manhattanDist(new_pos, food_pos))

        distance_weight = -1
        if min_food_dist != float('inf'):
            score += min_food_dist * distance_weight
        else:
            score += 5

        # 남은 food 개수 반영
        food_left_penalty = -1
        score += len(food_positions) * food_left_penalty

        return score


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    Common elements to all multi-agent searchers.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def is_terminal_state(self, game_state: GameState, depth):
        # 종료 조건 -> win / lose / depth 도달
        return game_state.isWin() or game_state.isLose() or (depth >= self.depth)

    def max_value(self, game_state: GameState, depth, agent_idx):
        # pacman 차례 -> 가능한 action 중 최대값 선택
        legal_actions = game_state.getLegalActions(agent_idx)

        if not legal_actions:
            return self.evaluationFunction(game_state), None

        best_value = float('-inf')
        best_action = None

        for action in legal_actions:
            next_state = game_state.generateSuccessor(agent_idx, action)

            # pacman 다음은 항상 ghost 1
            next_agent_idx = (agent_idx + 1) % game_state.getNumAgents()

            value, _ = self.value(next_state, depth, next_agent_idx)

            if value > best_value:
                best_value = value
                best_action = action

        return best_value, best_action

    def min_value(self, game_state: GameState, depth, agent_idx):
        # ghost 차례 -> 가능한 action 중 최소값 선택
        legal_actions = game_state.getLegalActions(agent_idx)

        if not legal_actions:
            return self.evaluationFunction(game_state), None

        best_value = float('inf')
        best_action = None

        for action in legal_actions:
            next_state = game_state.generateSuccessor(agent_idx, action)

            next_agent_idx = (agent_idx + 1) % game_state.getNumAgents()

            # 마지막 ghost가 끝나고 pacman으로 돌아올 때만 depth 증가
            if next_agent_idx == 0:
                value, _ = self.value(next_state, depth + 1, next_agent_idx)
            else:
                value, _ = self.value(next_state, depth, next_agent_idx)

            if value < best_value:
                best_value = value
                best_action = action

        return best_value, best_action

    def value(self, game_state: GameState, depth, agent_idx):
        # terminal state이면 evaluation function 값 반환
        if self.is_terminal_state(game_state, depth):
            return self.evaluationFunction(game_state), None

        if agent_idx == 0:
            return self.max_value(game_state, depth, agent_idx)
        else:
            return self.min_value(game_state, depth, agent_idx)

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """
        _, best_action = self.value(gameState, 0, 0)
        return best_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def is_terminal_state(self, game_state: GameState, depth):
        # 종료 조건 -> win / lose / depth 도달
        return game_state.isWin() or game_state.isLose() or (depth >= self.depth)

    def max_value(self, game_state: GameState, depth, alpha, beta):
        # pacman 차례 -> 가능한 action 중 최대값 선택
        legal_actions = game_state.getLegalActions(0)

        if not legal_actions:
            return self.evaluationFunction(game_state), None

        best_value = float('-inf')
        best_action = None

        for action in legal_actions:
            next_state = game_state.generateSuccessor(0, action)

            # pacman 다음은 항상 ghost 1
            next_agent_idx = 1

            value, _ = self.value(next_state, depth, next_agent_idx, alpha, beta)

            if value > best_value:
                best_value = value
                best_action = action

            alpha = max(alpha, best_value)

            # pruning
            if best_value > beta:
                return best_value, best_action

        return best_value, best_action

    def min_value(self, game_state: GameState, depth, agent_idx, alpha, beta):
        # ghost 차례 -> 가능한 action 중 최소값 선택
        legal_actions = game_state.getLegalActions(agent_idx)

        if not legal_actions:
            return self.evaluationFunction(game_state), None

        num_agents = game_state.getNumAgents()
        best_value = float('inf')
        best_action = None

        for action in legal_actions:
            next_state = game_state.generateSuccessor(agent_idx, action)

            next_agent_idx = (agent_idx + 1) % num_agents

            # 모든 ghost가 끝나고 pacman으로 돌아올 때만 depth 증가
            if next_agent_idx == 0:
                next_depth = depth + 1
            else:
                next_depth = depth

            value, _ = self.value(next_state, next_depth, next_agent_idx, alpha, beta)

            if value < best_value:
                best_value = value
                best_action = action

            beta = min(beta, best_value)

            # pruning
            if best_value < alpha:
                return best_value, best_action

        return best_value, best_action

    def value(self, game_state: GameState, depth, agent_idx, alpha, beta):
        # terminal state이면 evaluation function 값 반환
        if self.is_terminal_state(game_state, depth):
            return self.evaluationFunction(game_state), None

        if agent_idx == 0:
            return self.max_value(game_state, depth, alpha, beta)
        else:
            return self.min_value(game_state, depth, agent_idx, alpha, beta)

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        alpha = float('-inf')
        beta = float('inf')

        _, best_action = self.value(gameState, 0, 0, alpha, beta)
        return best_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def is_terminal_state(self, game_state: GameState, depth):
        # 종료 조건 -> win / lose / depth 도달
        return game_state.isWin() or game_state.isLose() or (depth >= self.depth)

    def max_value(self, game_state: GameState, depth, agent_idx):
        # pacman 차례 -> 가능한 action 중 최대값 선택
        legal_actions = game_state.getLegalActions(agent_idx)

        if not legal_actions:
            return self.evaluationFunction(game_state), None

        best_value = float('-inf')
        best_action = None

        for action in legal_actions:
            next_state = game_state.generateSuccessor(agent_idx, action)

            # pacman 다음은 항상 ghost 1
            next_agent_idx = (agent_idx + 1) % game_state.getNumAgents()

            value, _ = self.value(next_state, depth, next_agent_idx)

            if value > best_value:
                best_value = value
                best_action = action

        return best_value, best_action

    def exp_value(self, game_state: GameState, depth, agent_idx):
        # ghost 차례 -> 가능한 action들의 기대값 계산
        legal_actions = game_state.getLegalActions(agent_idx)

        if not legal_actions:
            return self.evaluationFunction(game_state), None

        expected_value = 0.0
        action_prob = 1.0 / len(legal_actions)

        for action in legal_actions:
            next_state = game_state.generateSuccessor(agent_idx, action)

            next_agent_idx = (agent_idx + 1) % game_state.getNumAgents()

            # 마지막 ghost가 끝나고 pacman으로 돌아올 때만 depth 증가
            if next_agent_idx == 0:
                value, _ = self.value(next_state, depth + 1, next_agent_idx)
            else:
                value, _ = self.value(next_state, depth, next_agent_idx)

            expected_value += action_prob * value

        return expected_value, None

    def value(self, game_state: GameState, depth, agent_idx):
        # terminal state이면 evaluation function 값 반환
        if self.is_terminal_state(game_state, depth):
            return self.evaluationFunction(game_state), None

        if agent_idx == 0:
            return self.max_value(game_state, depth, agent_idx)
        else:
            return self.exp_value(game_state, depth, agent_idx)

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        _, best_action = self.value(gameState, 0, 0)
        return best_action


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: 현재 score를 기본으로 사용하고,
    food는 가까운 거리와 남은 개수를 함께 고려했다.
    active ghost는 가까울수록 강하게 회피하고,
    scared ghost는 잡을 수 있을 때만 추격한다.
    capsule은 위험한 상황에서 더 높은 우선순위를 갖도록 설계했다.
    """
    "*** YOUR CODE HERE ***"

    if currentGameState.isWin():
        return 1000000
    if currentGameState.isLose():
        return -1000000

    current_pos = currentGameState.getPacmanPosition()
    food_list = currentGameState.getFood().asList()
    capsule_list = currentGameState.getCapsules()
    ghost_states = currentGameState.getGhostStates()

    score = currentGameState.getScore()

    "food 관련"
    if food_list:
        food_dist_list = [util.manhattanDist(current_pos, food_pos) for food_pos in food_list]
        closest_food_dist = min(food_dist_list)

        # 가까운 food 여러 개까지 같이 고려
        nearest_food_cnt = min(3, len(food_dist_list))
        avg_nearest_food_dist = sum(sorted(food_dist_list)[:nearest_food_cnt]) / nearest_food_cnt

        score += 12.0 / (closest_food_dist + 1)
        score += 6.0 / (avg_nearest_food_dist + 1)
        score -= 5.0 * len(food_list)
    else:
        score += 500.0

    "ghost 관련"
    alive_ghosts = [ghost for ghost in ghost_states if ghost.scaredTimer == 0]
    scared_ghosts = [ghost for ghost in ghost_states if ghost.scaredTimer > 0]

    "active ghost 관련"
    if alive_ghosts:
        alive_ghost_dist_list = [util.manhattanDist(current_pos, ghost.getPosition()) for ghost in alive_ghosts]
        closest_alive_ghost_dist = min(alive_ghost_dist_list)

        if closest_alive_ghost_dist <= 1:
            score -= 1200.0
        elif closest_alive_ghost_dist == 2:
            score -= 500.0
        elif closest_alive_ghost_dist == 3:
            score -= 180.0
        elif closest_alive_ghost_dist <= 5:
            score -= 60.0 / closest_alive_ghost_dist
        else:
            score -= 10.0 / closest_alive_ghost_dist

        # 여러 ghost가 동시에 가까우면 추가 페널티
        close_ghost_cnt = sum(1 for ghost_dist in alive_ghost_dist_list if ghost_dist <= 3)
        score -= 80.0 * max(0, close_ghost_cnt - 1)

    "scared ghost 관련"
    if scared_ghosts:
        for ghost in scared_ghosts:
            scared_ghost_dist = util.manhattanDist(current_pos, ghost.getPosition())
            scared_time = ghost.scaredTimer

            # 잡을 시간이 충분할 때만 scared ghost 추격
            if scared_time > scared_ghost_dist:
                score += 140.0 / (scared_ghost_dist + 1)
                score += min(scared_time - scared_ghost_dist, 6) * 5.0
            else:
                # 시간이 얼마 안 남았으면 무리해서 쫓지 않음
                if scared_ghost_dist <= 2:
                    score -= 25.0

    "capsule 관련"
    if capsule_list:
        capsule_dist_list = [util.manhattanDist(current_pos, capsule_pos) for capsule_pos in capsule_list]
        closest_capsule_dist = min(capsule_dist_list)

        if alive_ghosts:
            closest_alive_ghost_dist = min(
                util.manhattanDist(current_pos, ghost.getPosition()) for ghost in alive_ghosts
            )

            # active ghost가 가까울수록 capsule 우선순위를 높임
            if closest_alive_ghost_dist <= 3:
                score += 160.0 / (closest_capsule_dist + 1)
            elif closest_alive_ghost_dist <= 6:
                score += 80.0 / (closest_capsule_dist + 1)
            else:
                score += 30.0 / (closest_capsule_dist + 1)
        else:
            score += 15.0 / (closest_capsule_dist + 1)

        score -= 18.0 * len(capsule_list)

    "이동 가능성 관련"
    legal_moves = currentGameState.getLegalActions(0)
    movable_actions = [action for action in legal_moves if action != Directions.STOP]

    score += 4.0 * len(movable_actions)

    if len(movable_actions) <= 1 and alive_ghosts:
        score -= 100.0

    "모든 ghost가 scared 상태면 조금 더 공격적으로 평가"
    if scared_ghosts and not alive_ghosts:
        score += 60.0

    return score


# Abbreviation
better = betterEvaluationFunction
