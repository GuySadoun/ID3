from operator import itemgetter

from sklearn.model_selection import KFold

from ID3 import Node, entropy_h
import numpy as np
import pandas as pan


def dynamic_partition(column, node):
    # start = time.time()

    sorted_ascending_f_val = sorted(node.example_set, key=itemgetter(column))
    best_threshold = -1
    best_ig = 0
    set_size = len(sorted_ascending_f_val)
    best_ig_assigned = False
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
            best_ig_assigned = True
            best_threshold = optional_threshold
            best_h_e_1 = h_e_1
            best_h_e_2 = h_e_2
    assert best_ig_assigned
    part_left_set = sorted_ascending_f_val[:best_i]
    part_right_set = sorted_ascending_f_val[best_i:]

    # end = time.time()
    # print(f'dynamic_partition took {end - start} second')
    return best_ig, best_threshold, part_left_set, part_right_set, \
           best_h_e_1, best_h_e_2


def analyze_node_set(node_set):
    m_b_count = {'B': 0, 'M': 0}
    for example in node_set:
        m_b_count[example[0]] += 1
    default = 'M' if m_b_count['M'] > m_b_count['B'] else 'B'
    is_leaf = (m_b_count['B'] == 0 or m_b_count['M'] == 0)
    classification = None
    if is_leaf:
        if m_b_count['M'] == 0:
            classification = 'B'
        elif m_b_count['B'] == 0:
            classification = 'M'
        else:
            assert False
    return is_leaf, classification, default


def is_partition_valid(set_l, set_r, default_l, default_r, frc):
    if len(set_l) < frc*len(set_r):
        if default_r == 'M':
            return False
    else:
        if default_l == 'M':
            if len(set_r) < frc*len(set_l):
                return False
    return True


class CostSensitiveId3:
    nodes_in_tree = 0
    feature_to_column = {}

    def __init__(self, features_list, train_set_np):
        # start = time.time()

        self.is_tree_initialize = False
        m_b_count = {'B': 0, 'M': 0}
        self.features_list = features_list
        self.train_np = train_set_np
        for obj in self.train_np:
            m_b_count[obj[0]] += 1
        self.train_size = len(self.train_np)
        ratio_ill = m_b_count['M'] / self.train_size
        ratio_healthy = m_b_count['B'] / self.train_size
        is_leaf = (m_b_count['M'] == 0 or m_b_count['B'] == 0)
        default = 'M' if m_b_count['M'] > m_b_count['B'] else 'B'
        classification = None
        if is_leaf:
            classification = 'B' if m_b_count['M'] == 0 else 'M'
        starting_entropy = -(ratio_ill * np.math.log(ratio_ill, 2) + ratio_healthy * np.math.log(ratio_healthy, 2))
        self.head = Node(self.train_np, starting_entropy, is_leaf, self.nodes_in_tree, classification,
                         default=default)
        self.nodes_in_tree += 1

        # Q4.3 addition ##
        R_val = [0.05, 0.1, 0.15, 0.2]
        best_frac = R_val[0]
        minimum_loss = 0
        self.K_fold = KFold(n_splits=5, shuffle=True, random_state=303026462)
        loss_list = []
        false_negative_t = false_positive_t = 0
        for frac in R_val:
            loss_res_for_frac = []
            for train_index, test_index in self.K_fold.split(self.train_np):
                X_train, X_test = self.train_np[train_index], self.train_np[test_index]
                m_b_t_count = {'B': 0, 'M': 0}
                train_size = len(X_train)
                for obj in X_train:
                    m_b_t_count[obj[0]] += 1
                ill_ratio = m_b_t_count['M'] / train_size
                healthy_ratio = m_b_t_count['B'] / train_size
                is_t_leaf = (m_b_t_count['M'] == 0 or m_b_t_count['B'] == 0)
                default_t = 'M' if m_b_t_count['M'] >= m_b_t_count['B'] else 'B'
                classification_t = None
                if is_t_leaf:
                    classification_t = 'B' if m_b_count['M'] == 0 else 'M'
                starting_t_entropy = -(ill_ratio * np.math.log(ill_ratio, 2) + healthy_ratio * np.math.log(healthy_ratio, 2))
                head = Node(X_train, starting_t_entropy, is_t_leaf, 0, classification_t, default=default_t)
                self.__create_classifier_tree(head, ratio=frac)
                test_set_size_t = len(X_test)
                for test_subj in X_test:
                    real_diagnosis_t = test_subj[0]
                    diagnosis_t = self.classifier(test_subj, head)
                    if diagnosis_t != real_diagnosis_t:
                        if diagnosis_t == 'B':
                            false_negative_t += 1
                        else:
                            assert diagnosis_t == 'M'
                            false_positive_t += 1
                loss_res_for_frac.append((0.1 * false_positive_t + false_negative_t) / test_set_size_t)
            loss_list.append(sum(loss_res_for_frac)/self.K_fold.get_n_splits())
        for i in range(len(loss_list)):
            if loss_list[i] < minimum_loss:
                best_frac = R_val[i]
                print(f'best_frac = {best_frac}')
        self.__create_classifier_tree(ratio=best_frac)

        # end = time.time()
        # print(f' __init__ took {end - start} second')

    def __create_classifier_tree(self, node=None, ratio=0.0, is_train=False):
        # start = time.time()
        init = False
        if node is None:
            self.is_tree_initialize = True
            init = True
            node = self.head
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
        column = 0
        for feature in self.features_list:
            if init:
                self.feature_to_column[feature] = column
            if feature != 'diagnosis':
                ig, threshold, set_l, set_r, h_e_l, h_e_r = dynamic_partition(column, node)
                if ig > best_dict["ig"]:
                    best_dict["ig"] = ig
                    best_dict["threshold"] = threshold
                    best_dict["set_l"] = set_l
                    best_dict["set_r"] = set_r
                    best_dict["h_e_l"] = h_e_l
                    best_dict["h_e_r"] = h_e_r
                    selected_feature = feature
            column += 1
        # assert selected_feature is not None
        is_l_leaf, classification_l, default_l = analyze_node_set(best_dict["set_l"])
        is_r_leaf, classification_r, default_r = analyze_node_set(best_dict["set_r"])

        if not is_partition_valid(best_dict["set_l"], best_dict["set_r"], default_l, default_r, ratio):
            node.set_as_a_leaf()
            return

        node.set_separator(selected_feature, best_dict["threshold"])
        left_son = Node(best_dict["set_l"], best_dict["h_e_l"], is_l_leaf, self.nodes_in_tree,
                        leaf_diagnosis=classification_l, father=node, default=default_l)
        self.nodes_in_tree += 1
        right_son = Node(best_dict["set_r"], best_dict["h_e_r"], is_r_leaf, self.nodes_in_tree,
                         leaf_diagnosis=classification_r, father=node, default=default_r)
        if not is_train:
            self.nodes_in_tree += 1
        node.set_left_son(left_son)
        node.set_right_son(right_son)
        self.__create_classifier_tree(left_son, ratio=ratio, is_train=is_train)
        self.__create_classifier_tree(right_son, ratio=ratio, is_train=is_train)

        # end = time.time()
        # print(f' create_classifier_tree took {end - start} second')

    def classifier(self, subject, root=None):
        if not self.is_tree_initialize:
            self.__create_classifier_tree()
        is_pandas_line = isinstance(subject, pan.Series)
        node = self.head if root is None else root
        while not node.is_leaf:
            feature = node.separating_f
            key = feature if is_pandas_line else self.feature_to_column[feature]
            subj_f_val = subject[key]
            if subj_f_val >= node.threshold:
                next_node = node.right_son
                alternative = node.left_son
            else:
                next_node = node.left_son
                alternative = node.right_son
            # Q4.3 addition ##
            if next_node.is_leaf and next_node.leaf_diagnosis == 'B' \
                    and abs(subj_f_val - node.threshold)/node.threshold < 0.1:
                node = alternative
            else:
                node = next_node
        # assert node.is_leaf
        # if node.leaf_diagnosis == 'B' and subject[0] == 'M':
        #     print(f'false negative - id: {node.node_id}')
        #     print(f'father th: {node.father.threshold} of feature: {node.father.separating_f}')
        #     print(f'current f_val - id: {subj_f_val}')
        #     print(f'percent of misvalue = {abs(subj_f_val-node.father.threshold)/node.father.threshold}')

        return node.leaf_diagnosis


if __name__ == '__main__':
    train_set = pan.read_csv('/home/guy-pc/PycharmProjects/ID3/train.csv')
    test_set = pan.read_csv('/home/guy-pc/PycharmProjects/ID3/test.csv')
    features = train_set.keys()
    train_np = train_set.to_numpy()
    cost_sensitive = CostSensitiveId3(features, train_np)
    correct_answers = 0
    test_set_size = test_set['diagnosis'].size
    false_negative = false_positive = 0
    for subj in test_set.iloc:
        diagnosis = cost_sensitive.classifier(subj)
        real_diagnosis = subj['diagnosis']
        if diagnosis == real_diagnosis:
            correct_answers += 1
        else:
            if diagnosis == 'B':
                false_negative += 1
            else:
                assert diagnosis == 'M'
                false_positive += 1
    # print(f'false_positive: {false_positive}, false_negative: {false_negative}')
    # print(f'correct/tested = {correct_answers}/{test_set_size}')
    print(f'{(0.1 * false_positive + false_negative) / test_set_size}')
