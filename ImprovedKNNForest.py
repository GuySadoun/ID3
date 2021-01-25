from math import sqrt
from operator import itemgetter
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pan
from ID3 import entropy_h, Node
from KNNForest import KNNForest


def dynamic_partition(column, node):
    sorted_ascending_f_val = sorted(node.example_set, key=itemgetter(column))
    best_threshold = -1
    best_ig = 0
    set_size = len(sorted_ascending_f_val)
    best_h_e_1 = best_h_e_2 = 0
    best_i = 1

    for i in range(0, set_size - 1):
        i_f_val = sorted_ascending_f_val[i][column]
        next_f_val = sorted_ascending_f_val[i + 1][column]
        optional_threshold = (i_f_val + next_f_val) / 2
        smaller_then_threshold = sorted_ascending_f_val[:i + 1]
        bigger_then_threshold = sorted_ascending_f_val[i + 1:]
        h_e_1 = entropy_h(smaller_then_threshold)
        ratio_1 = (i + 1) / set_size
        h_e_2 = entropy_h(bigger_then_threshold)
        ratio_2 = (set_size - 1 - i) / set_size
        partition_entropy = ratio_1 * h_e_1 + ratio_2 * h_e_2
        ig = node.entropy - partition_entropy
        if ig > best_ig:
            best_i = i + 1
            best_ig = ig
            best_threshold = optional_threshold
            best_h_e_1 = h_e_1
            best_h_e_2 = h_e_2

    part_left_set = sorted_ascending_f_val[:best_i]
    part_right_set = sorted_ascending_f_val[best_i:]

    return best_ig, best_threshold, part_left_set, part_right_set, \
           best_h_e_1, best_h_e_2


def analyze_node_set(node_set):
    m_b_count = {'B': 0, 'M': 0}
    for example in node_set:
        m_b_count[example[0]] += 1
    default = 'M' if m_b_count['M'] >= m_b_count['B'] else 'B'
    is_leaf = m_b_count['B'] == 0 or m_b_count['M'] == 0
    classification = None
    if is_leaf:
        if m_b_count['M'] == 0:
            classification = 'B'
        elif m_b_count['B'] == 0:
            classification = 'M'
        else:
            assert False
    return is_leaf, classification, default


class ImprovedKNNForest:
    feature_to_index = {}
    index_to_feature = {}

    def __init__(self, features_list, train_np, committee_size, participating_trees):
        if committee_size < participating_trees:
            print('can\'t choose more then I have, all trees will be the committee for all decisions')
        self.committee_size_N = committee_size
        self.participating_trees_K = participating_trees
        self.features_list = features_list
        self.train_np = train_np
        self.train_size = len(self.train_np)
        self.committee_trees = {}
        column = 0
        for feature in self.features_list:
            self.feature_to_index[feature] = column
            self.index_to_feature[column] = feature
            column += 1
        root_id = 0
        train_list = train_np.tolist()
        for i in range(self.committee_size_N):
            p = random.uniform(0.3, 0.7)
            sample_size = int(np.ceil(p * self.train_size))
            train_sample = random.sample(train_list, sample_size)
            m_b_count = {'B': 0, 'M': 0}
            for obj in train_sample:
                m_b_count[obj[0]] += 1
            ratio_ill = m_b_count['M'] / sample_size
            ratio_healthy = m_b_count['B'] / sample_size
            is_leaf = (m_b_count['M'] == 0 or m_b_count['B'] == 0)
            default = 'M' if m_b_count['M'] >= m_b_count['B'] else 'B'
            classification = None
            if is_leaf:
                classification = 'B' if m_b_count['M'] == 0 else 'M'
            root_entropy = -(ratio_ill * np.math.log(ratio_ill, 2) + ratio_healthy * np.math.log(ratio_healthy, 2))
            # Q7 changes: first we find the dividing features, than we create the centroid with those features
            self.committee_trees[root_id] = self.Tree(train_sample, root_entropy, is_leaf, classification, default)

            # NOTICE: uncomment this part for calc dist(as explained in dry part) only with separating features
            # self.__create_decision_tree(self.committee_trees[root_id].root,
            #                             self.committee_trees[root_id].value_features)

            # NOTICE: uncomment this part for calc dist(as explained in dry part) with all features
            self._create_decision_tree(self.committee_trees[root_id].root)
            self.committee_trees[root_id].value_features = features_list[1:]

            centroid = self.create_centroid(train_sample, self.committee_trees[root_id].value_features)
            self.committee_trees[root_id].set_centroid(centroid)
            root_id += 1

    def __create_decision_tree(self, node, features_list):
        if node.is_leaf:  # stop condition
            return
        selected_feature = None
        best_dict = {
            "ig": 0,
            "threshold": 0,
            "set_l": [],
            "set_r": [],
            "h_e_l": 1,
            "h_e_r": 1
        }

        for feature in self.features_list:
            if feature != 'diagnosis':
                ig, threshold, set_l, set_r, h_e_l, h_e_r = dynamic_partition(self.feature_to_index[feature], node)
                if ig > best_dict["ig"]:
                    best_dict["ig"] = ig
                    best_dict["threshold"] = threshold
                    best_dict["set_l"] = set_l
                    best_dict["set_r"] = set_r
                    best_dict["h_e_l"] = h_e_l
                    best_dict["h_e_r"] = h_e_r
                    selected_feature = feature
        features_list.append(selected_feature)  # for only separating features included in dist calculation
        is_l_leaf, classification_l, default_l = analyze_node_set(best_dict["set_l"])
        is_r_leaf, classification_r, default_r = analyze_node_set(best_dict["set_r"])

        node.set_separator(selected_feature, best_dict["threshold"])
        left_son = Node(best_dict["set_l"], best_dict["h_e_l"], is_l_leaf, -1,
                        leaf_diagnosis=classification_l, father=node, default=default_l)
        right_son = Node(best_dict["set_r"], best_dict["h_e_r"], is_r_leaf, -1,
                         leaf_diagnosis=classification_r, father=node, default=default_r)
        node.set_left_son(left_son)
        node.set_right_son(right_son)
        self.__create_decision_tree(left_son, features_list)
        self.__create_decision_tree(right_son, features_list)

    def _create_decision_tree(self, node):
        if node.is_leaf:  # stop condition
            return
        selected_feature = None
        best_dict = {
            "ig": 0,
            "threshold": 0,
            "set_l": [],
            "set_r": [],
            "h_e_l": 1,
            "h_e_r": 1
        }

        for feature in self.features_list:
            if feature != 'diagnosis':
                ig, threshold, set_l, set_r, h_e_l, h_e_r = dynamic_partition(self.feature_to_index[feature], node)
                if ig > best_dict["ig"]:
                    best_dict["ig"] = ig
                    best_dict["threshold"] = threshold
                    best_dict["set_l"] = set_l
                    best_dict["set_r"] = set_r
                    best_dict["h_e_l"] = h_e_l
                    best_dict["h_e_r"] = h_e_r
                    selected_feature = feature
        is_l_leaf, classification_l, default_l = analyze_node_set(best_dict["set_l"])
        is_r_leaf, classification_r, default_r = analyze_node_set(best_dict["set_r"])

        node.set_separator(selected_feature, best_dict["threshold"])
        left_son = Node(best_dict["set_l"], best_dict["h_e_l"], is_l_leaf, -1,
                        leaf_diagnosis=classification_l, father=node, default=default_l)
        right_son = Node(best_dict["set_r"], best_dict["h_e_r"], is_r_leaf, -1,
                         leaf_diagnosis=classification_r, father=node, default=default_r)
        node.set_left_son(left_son)
        node.set_right_son(right_son)
        self._create_decision_tree(left_son)
        self._create_decision_tree(right_son)

    def classifier(self, subject):
        is_pandas_line = isinstance(subject, pan.Series)
        # choose k trees by euclid dist from centroid, classify with each tree, return majority decision
        tree_relative_dist_tuple_set = []
        relative_dist = lambda x, y: (abs(x - y)) / ((x + y) / 2)
        if self.participating_trees_K < self.committee_size_N:
            for tree_id, tree in self.committee_trees.items():
                sum_diff = 0
                centroid = tree.get_centroid()
                features_to_measure = tree.value_features
                for centroid_index, feature in enumerate(features_to_measure):
                    key = feature if is_pandas_line else self.feature_to_index[feature]
                    sum_diff += relative_dist(centroid[centroid_index], subject[key])
                euclid_dist = sqrt(sum_diff)
                tree_relative_dist_tuple_set.append((tree_id, euclid_dist))
            tree_relative_dist_tuple_set.sort(key=lambda tup: tup[1])
        else:  # if K >= N don't waste time on calculate euclid dist
            tree_relative_dist_tuple_set = [(x, 0) for x in range(self.committee_size_N)]
        votes = {'B': 0, 'M': 0}
        for index_dist_tup in tree_relative_dist_tuple_set[:self.participating_trees_K]:  # K trees in committee
            node = self.committee_trees[index_dist_tup[0]].root  # classify with each tree
            while not node.is_leaf:
                feature = node.separating_f
                key = feature if is_pandas_line else self.feature_to_index[feature]
                subj_f_val = subject[key]
                if subj_f_val >= node.threshold:
                    node = node.right_son
                else:
                    node = node.left_son
            votes[node.leaf_diagnosis] += 1
        majority = 'B' if votes['B'] > votes['M'] else 'M'  # return majority decision
        return majority

    def create_centroid(self, example_set, features_list):
        centroid = []
        for f in features_list:
            centroid.append(example_set[0][self.feature_to_index[f]])
        example_set_size = len(example_set)
        for example in example_set[1:]:
            for i, f in enumerate(features_list):
                centroid[i] += example[self.feature_to_index[f]]
        centroid = map(lambda x: x / example_set_size, centroid)
        return tuple(centroid)

    class Tree:
        centroid: tuple
        is_centroid_init = False

        def __init__(self, e_set, entropy, is_leaf, leaf_diagnosis=None, default=None):
            self.root = Node(e_set, entropy, is_leaf, -1, leaf_diagnosis, None, None, 0, default)
            self.value_features = []

        def set_centroid(self, centroid):
            self.centroid = centroid
            self.is_centroid_init = True

        def get_centroid(self):
            if self.is_centroid_init:
                return self.centroid
            else:
                return None


if __name__ == '__main__':
    train_set = pan.read_csv('/home/guy-pc/PycharmProjects/ID3/train.csv')
    test_set = pan.read_csv('/home/guy-pc/PycharmProjects/ID3/test.csv')
    features = train_set.keys()
    train_set_np = train_set.to_numpy()
    test_set_size = len(test_set.index)
    # accuracy_list = []
    # for k in [5, 7, 9, 11, 13]:
    #     for n in range(k, 18, 2):
    #         improved_knn_forest = ImprovedKNNForest(features, train_set_np, n, k)
    #         correct_answers = 0
    #         for subj in test_set.iloc:
    #             diagnosis = improved_knn_forest.classifier(subj)
    #             real_diagnosis = subj['diagnosis']
    #             if diagnosis == real_diagnosis:
    #                 correct_answers += 1
    #         accuracy_list.append((n, k, correct_answers / test_set_size))
    # accuracy_list.sort(key=lambda tup: tup[2], reverse=True)
    # print(accuracy_list[0][2])

    # prove of improvement:
    accuracy_list = []
    for k in [5, 7, 9, 11, 13]:
        for n in range(k, 18, 2):
            improved_knn_forest = ImprovedKNNForest(features, train_set_np, n, k)
            knn_forest = KNNForest(features, train_set_np, n, k)
            correct_answers_improved = correct_answers_knn = 0
            for subj in test_set.iloc:
                real_diagnosis = subj['diagnosis']
                diagnosis_improved_knn = improved_knn_forest.classifier(subj)
                diagnosis_knn = knn_forest.classifier(subj)
                if diagnosis_improved_knn == real_diagnosis:
                    correct_answers_improved += 1
                if diagnosis_knn == real_diagnosis:
                    correct_answers_knn += 1
            accuracy_list.append((n, k, correct_answers_improved / test_set_size, correct_answers_knn / test_set_size))
    better_accuracy = {'KNN': 0, 'ImprovedKNN': 0, 'same': 0}
    for tup in accuracy_list:
        # print(f'for N={tup[0]}, K={tup[1]}, regular KNN accuracy = {tup[2]}, improved = {tup[3]}')
        if tup[3] > tup[2]:
            better_accuracy['ImprovedKNN'] += 1
        elif tup[3] < tup[2]:
            better_accuracy['KNN'] += 1
        else:
            better_accuracy['same'] += 1
    print(f'Improved was better {better_accuracy["ImprovedKNN"]} times, KNN was better {better_accuracy["KNN"]}, same - {better_accuracy["same"]}')
