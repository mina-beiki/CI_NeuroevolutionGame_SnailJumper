import copy
from operator import attrgetter

from player import Player
import numpy as np


class Evolution:
    def __init__(self):
        self.game_mode = "Neuroevolution"

    def next_population_selection(self, players, num_players):
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        """
        # TODO (Implement top-k algorithm here)
        players_sorted = sorted(players, key=lambda player: player.fitness, reverse=True)
        min_fitness = players_sorted[len(players_sorted) - 1].fitness
        max_fitness = players_sorted[0].fitness
        fitness_list = [p.fitness for p in players]
        mean_fitness = sum(fitness_list) / len(fitness_list)
        players = players_sorted
        f = open("data.txt", 'a')
        f.write(f"{max_fitness} {min_fitness} {mean_fitness} \n")
        return players[: num_players]

        # TODO (Additional: Implement roulette wheel here)
        # return self.roulette_wheel(players, num_players)

        # TODO (Additional: Implement SUS here)
        # return self.sus(players, num_players)

        # TODO (Additional: Implement SUS here)
        # return self.q_tournament(players, num_players, 0.5)

        # TODO (Additional: Learning curve)

    def roulette_wheel(self, players, num_players):
        next_generation = []
        fit_sum = sum([player.fitness for player in players])
        probs = [player.fitness / fit_sum for player in players]
        for i in range(num_players):
            p = np.random.choice(players, 1, p=probs)
            next_generation.append(p)

        return next_generation

    def sus(self, players, num_players):
        probas = []
        sum_fitness = 0
        for player in players:
            sum_fitness += player.fitness
        for player in players:
            probas.append(player.fitness / sum_fitness)
        for i in range(1, len(players)):
            probas[i] += probas[i - 1]

        random_number = np.random.uniform(0, 1 / num_players, 1)
        step = (probas[len(probas) - 1] - random_number) / num_players
        results = []

        for i in range(num_players):
            now = (i + 1) * step
            for i, proba in enumerate(probas):
                if now <= proba:
                    results.append(self.clone_player(players[i]))
                    break
        return results

    def q_tournament(self, players, num_players, q):
        selected = []
        for i in range(num_players):
            q_selections = np.random.choice(players, q)
            selected.append(max(q_selections, key=attrgetter('fitness')))

        return selected

    def generate_new_population(self, num_players, prev_players=None):
        """
        Gets survivors and returns a list containing num_players number of children.

        :param num_players: Length of returning list
        :param prev_players: List of survivors
        :return: A list of children
        """
        first_generation = prev_players is None
        if first_generation:
            return [Player(self.game_mode) for _ in range(num_players)]
        else:
            # TODO ( Parent selection and child generation )
            prev_parents = prev_players.copy()
            new_players = []
            # roulette wheel:
            # prev_parents = self.roulette_wheel(prev_parents, len(prev_parents))
            # sus:
            prev_parents = self.sus(prev_parents, len(prev_parents))
            # q tournament:
            # prev_parents = self.q_tournament(prev_parents ,num_players ,3)

            # crossover:
            for i in range(0, len(prev_parents), 2):
                child1, child2 = self.child_production(prev_parents[i], prev_parents[i + 1])
                new_players.append(child1)
                new_players.append(child2)

            return new_players

    def child_production(self, parent1, parent2):
        child1 = self.clone_player(parent1)
        child2 = self.clone_player(parent2)

        self.crossover(child1.nn.W_1, child2.nn.W_1, parent1.nn.W_1, parent2.nn.W_1)
        self.crossover(child1.nn.W_2, child2.nn.W_2, parent1.nn.W_2, parent2.nn.W_2)
        self.crossover(child1.nn.b_1, child2.nn.b_1, parent1.nn.b_1, parent2.nn.b_1)
        self.crossover(child1.nn.b_2, child2.nn.b_2, parent1.nn.b_2, parent2.nn.b_2)

        # print("AFTER ", child1.nn.W_1 - tmp)
        self.mutate(child1)
        self.mutate(child2)
        return child1, child2

    def clone_player(self, player):
        """
        Gets a player as an input and produces a clone of that player.
        """
        new_player = Player(self.game_mode)
        new_player.nn = copy.deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player

    def crossover(self, child1, child2, parent1, parent2):
        row_size, column_size = child1.shape
        section_1, section_2 = int(row_size / 3), int(2 * row_size / 3)

        random_number = np.random.uniform(0, 1, 1)

        if random_number > 0.5:
            child1[:section_1, :] = parent1[:section_1:, :]
            child1[section_1:section_2, :] = parent2[section_1:section_2, :]
            child1[section_2:, :] = parent1[section_2:, :]

            child2[:section_1, :] = parent2[:section_1:, :]
            child2[section_1:section_2, :] = parent1[section_1:section_2, :]
            child2[section_2:, :] = parent2[section_2:, :]
        else:
            child1[:section_1, :] = parent2[:section_1:, :]
            child1[section_1:section_2, :] = parent1[section_1:section_2, :]
            child1[section_2:, :] = parent2[section_2:, :]

            child2[:section_1, :] = parent1[:section_1:, :]
            child2[section_1:section_2, :] = parent2[section_1:section_2, :]
            child2[section_2:, :] = parent1[section_2:, :]

    def mutate(self, child):
        # child: an object of class `Player`
        threshold = 0.3
        # random_number = np.random.uniform(0, 1, 1)
        # if random_number < threshold:
        self.add_noise(child.nn.W_1, threshold)
        self.add_noise(child.nn.W_2, threshold)
        self.add_noise(child.nn.b_1, threshold)
        self.add_noise(child.nn.b_2, threshold)

    def add_noise(self, array, threshold):
        random_number = np.random.uniform(0, 1, 1)
        if random_number < threshold:
            array += np.random.randn(array.shape[0] * array.shape[1]).reshape(array.shape[0], array.shape[1])
