import numpy as np
import sys
from iForest_extended import Copy_Node, Node
from tqdm import trange


class EEIF(object):
    """docstring for EEIF"""

    def __init__(
            self,
            data,
            labels,
            iteration,
            threshold
    ):
        super(EEIF, self).__init__()
        self.labels = labels
        self.iteration = iteration
        self.threshold = threshold
        self.sub_sample_size = min(len(data), 256)
        self.num_tree = 100
        self.lr = 1 / np.sqrt(self.num_tree)
        self.ls = np.ceil(np.log2(self.sub_sample_size))
        self.sigma = {}
        self.bound = []
        self.forest = {}

        data_dim = data.shape[1]
        sigmas = [1 / 4 / np.sqrt(self.num_tree) * np.exp(np.random.randn()) for _ in range(self.num_tree)]
        forest = []
        bounds = []

        # 1: building trees
        for _ in range(self.num_tree):
            sub_data = data[np.random.choice(len(data), self.sub_sample_size, replace=False)]
            sub_data = np.array(sub_data, dtype=float)
            bound = np.array(
                [np.amin(sub_data, axis=0), np.amax(sub_data, axis=0)],
                dtype=np.float64
            ).T
            bounds.append(bound)
            forest.append(Node(sub_data, bound, 0, self.ls, data_dim, data_dim-1))               # check extension level

        scoress, avg_scores, = \
            [], []

        # Compute baseline scores of the first forest
        for pt in data:
            avg_score, scores = self.compute_score(pt, forest, self.sub_sample_size)
            scoress.append(scores)
            avg_scores.append(avg_score)

        # Select negative and positive labels
        pos_idx = np.where(labels)[0]
        neg_idx = np.where(labels == 0)[0]

        pos_features = data[pos_idx]
        neg_features = data[neg_idx]

        # The fitness function
        def fitness_fuc(tree):
            score_neg = []
            for pt in neg_features:
                score_neg.append(2 ** (-tree.path_len(pt, 0) / tree.avg_external_len(self.sub_sample_size)))
            score_neg = np.array(score_neg)

            score_pos = []
            for pt in pos_features:
                score_pos.append(2 ** (-tree.path_len(pt, 0) / tree.avg_external_len(self.sub_sample_size)))
            score_pos = np.array(score_pos)

            score_pos[score_pos > 0.75] = 0.75
            score_neg[score_neg < 0.1] = 0.1

            fitness = np.sum((score_pos - 0.75) ** 2) + np.sum((score_neg - 0.1) ** 2)
            fitness = - fitness
            return fitness

        # Update the trees for a given number of iteration by performing mutation and crossover operations
        if self.iteration > 0:
            for _ in trange(self.iteration):
                off_size = int(np.ceil(self.num_tree * 3))
                for num in range(off_size):
                    if np.random.uniform() < 0.8:
                        parents_idx = np.random.choice(self.num_tree, 2)
                        parents = [forest[idx] for idx in parents_idx]
                        sigma = sigmas[parents_idx[0]]
                        child = self.crossover(parents[0], parents[1], data)
                        rand_mask = np.random.randint(0, 2, size=bounds[0].shape, dtype=bool)
                        bound = bounds[parents_idx[0]] * rand_mask + bounds[parents_idx[1]] * (1 - rand_mask)
                        child, c_sigma = self.mutate(child, sigma, bound, data_dim)
                    else:
                        sub_data = data[np.random.choice(len(data), self.sub_sample_size, replace=False)]
                        sub_data = np.array(sub_data, dtype=float)
                        bound = np.array(
                            [np.amin(sub_data, axis=0), np.amax(sub_data, axis=0)],
                            dtype=np.float64
                        ).T
                        child = Node(sub_data, bound, 0, self.ls, data_dim, data_dim-1)          # check extension level
                        c_sigma = 1 / 4 / np.sqrt(self.num_tree) * np.exp(np.random.randn())

                    forest.append(child)
                    sigmas.append(c_sigma)

                # Select the best fitted trees for the new forest
                selected_idx, p_avg_fitness, avg_fitness = self.selection(forest, self.num_tree, fitness_fuc)

                forest = [forest[idx] for idx in selected_idx]
                sigmas = [sigmas[idx] for idx in selected_idx]

        self.forest = forest
        self.sigma = sigmas


    # The Mutation operation
    def mutate(self, indi, sig, bound, dim):
        child = Copy_Node(indi)
        sig = sig * np.exp(self.lr * np.random.randn())
        child.mutate(sig, bound, dim, dim-1)                                                     # check extension level
        return child, sig


    # The Crossover operation
    def crossover(self, p1, p2, data):
        p1c, p2c = Copy_Node(p1), Copy_Node(p2)
        crossover_pt2 = p2c.locate_crossover()
        if crossover_pt2 is not None:
            if_crossed = p1c.set_crossover(crossover_pt2)
            if if_crossed:
                p1c.train(data[np.random.choice(len(data), self.sub_sample_size, replace=False)])
        return p1c


    # Selects the best fitted trees
    def selection(self, population, pop_size, fitness_fuc):
        fitness = []
        for indi in population:
            fitness.append(fitness_fuc(indi))

        fitness = np.array(fitness)
        fit_rank = np.argsort(fitness)[::-1]
        selected = fit_rank[:pop_size]
        return selected, np.mean(fitness[:self.num_tree]), np.mean(fitness[selected])


    # Runs the dataset through the EEIF and reports anomaly scores back
    def run_full_test(self, data):
        prediction = []
        scores = []
        for pt in np.array(data, dtype=float):
            s, _ = self.compute_score(pt, self.forest, self.sub_sample_size)
            prediction.append(s > self.threshold)
            scores.append(s)
        return prediction, scores


    # Computes the anomaly score a data point gets in a forest
    def compute_score(self, pt, fores, sub_sample):
        sum_path_len = 0
        score_per_tree = []
        for tree in fores:
            avg_ext = tree.avg_external_len(sub_sample)
            if avg_ext == 0:
                sys.exit("The tree might have no nodes")

            path_len = tree.path_len(pt, 0)
            sum_path_len += path_len
            score_per_tree.append(path_len / avg_ext)

        if len(score_per_tree) == 0:
            return 0, None

        score_per_tree = np.array(score_per_tree)
        return 2 ** ((-sum_path_len) / self.num_tree / fores[0].avg_external_len(sub_sample)), 2 ** (-score_per_tree)
