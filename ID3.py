from operator import itemgetter
import numpy as np
import pandas as pan
import time


class Node:
    def __init__(self, e_set, entropy, is_leaf, node_id, leaf_diagnosis=None, father=None, separator=None,
                 threshold=0, default=None):
        self.example_set = e_set
        self.is_leaf = is_leaf
        self.node_id = node_id
        self.leaf_diagnosis = leaf_diagnosis
        self.father = father
        self.separating_f = separator
        self.right_son = self.left_son = None
        self.entropy = entropy
        self.threshold = threshold
        self.default = default

    def set_left_son(self, node):
        self.left_son = node

    def set_right_son(self, node):
        self.right_son = node

    def set_father(self, node):
        self.father = node

    def set_separator(self, feature, threshold):
        self.separating_f = feature
        self.threshold = threshold

    def set_as_a_leaf(self):
        self.is_leaf = True
        self.leaf_diagnosis = self.default

    def is_node_leaf(self):
        return self.is_leaf

    def print_node(self, with_sets=False):
        print('---------------------------')
        print(f'node ID: {self.node_id}')
        if self.father is not None:
            print(f'father = {self.father.node_id}')
        print(f'node set size = {len(self.example_set)}')
        if with_sets:
            print('node set: ')
            print(self.example_set)
        print(f'separating_f = {self.separating_f}')
        print(f'threshold = {self.threshold}')
        print(f'entropy = {self.entropy}')
        print(f'is_leaf = {self.is_leaf}')
        if self.is_leaf:
            print(f'leaf diagnosis = {self.leaf_diagnosis}')
        print('---------------------------')


def entropy_h(node_set):
    # start = time.time()

    healthy = sick = 0
    for line in node_set:
        if line[0] == 'B':
            healthy += 1
        elif line[0] == 'M':
            sick += 1
        else:
            assert False
    p_b = healthy / len(node_set)
    p_m = sick / len(node_set)
    h_b = p_b * np.math.log(p_b, 2) if p_b != 0 else 0
    h_m = p_m * np.math.log(p_m, 2) if p_m != 0 else 0

    # end = time.time()
    # print(f' entropy_h took {end - start} second')
    return -(h_b + h_m)


class ID3:
    is_tree_initialize: bool
    nodes_in_tree = 0

    def __init__(self, train_file_path, min_node_set_size=1):
        # start = time.time()

        self.min_node_size = min_node_set_size
        self.is_tree_initialize = False
        m_b_count = {'B': 0, 'M': 0}
        self.train_df = pan.read_csv(train_file_path)
        self.train_np = self.train_df.to_numpy()
        for row in self.train_df.iloc:
            m_b_count[row['diagnosis']] += 1
        self.train_size = self.train_df['diagnosis'].size
        ratio_ill = m_b_count['M'] / self.train_size
        ratio_healthy = m_b_count['B'] / self.train_size
        is_leaf = (m_b_count['M'] == 0 or m_b_count['B'] == 0) or (len(self.train_np) < self.min_node_size)
        default = 'M' if m_b_count['M'] > m_b_count['B'] else 'B'
        classification = None
        if is_leaf:
            classification = 'B' if m_b_count['M'] == 0 else 'M'
        starting_entropy = -(ratio_ill * np.math.log(ratio_ill, 2) + ratio_healthy * np.math.log(ratio_healthy, 2))
        self.head = Node(self.train_np, starting_entropy, is_leaf, self.nodes_in_tree, classification, default=default)
        self.nodes_in_tree += 1

        # end = time.time()
        # print(f' __init__ took {end - start} second')

    def create_classifier_tree(self, node=None):
        # start = time.time()

        if node is None:
            self.is_tree_initialize = True
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
        for feature in self.train_df.columns:
            if feature != 'diagnosis':
                is_valid, ig, threshold, set_l, set_r, h_e_l, h_e_r = self.dynamic_partition(column, node)
                if not is_valid:
                    return
                if ig > best_dict["ig"]:
                    best_dict["ig"] = ig
                    best_dict["threshold"] = threshold
                    best_dict["set_l"] = set_l
                    best_dict["set_r"] = set_r
                    best_dict["h_e_l"] = h_e_l
                    best_dict["h_e_r"] = h_e_r
                    selected_feature = feature
            column += 1
        assert selected_feature is not None

        is_l_leaf, classification_l, default_l = self.analyze_node_set(best_dict["set_l"])
        is_r_leaf, classification_r, default_r = self.analyze_node_set(best_dict["set_r"])

        node.set_separator(selected_feature, best_dict["threshold"])
        left_son = Node(best_dict["set_l"], best_dict["h_e_l"], is_l_leaf, self.nodes_in_tree,
                        leaf_diagnosis=classification_l, father=node, default=default_l)
        self.nodes_in_tree += 1
        right_son = Node(best_dict["set_r"], best_dict["h_e_r"], is_r_leaf, self.nodes_in_tree,
                         leaf_diagnosis=classification_r, father=node, default=default_r)
        self.nodes_in_tree += 1
        node.set_left_son(left_son)
        node.set_right_son(right_son)
        self.create_classifier_tree(left_son)
        self.create_classifier_tree(right_son)

        # end = time.time()
        # print(f' create_classifier_tree took {end - start} second')

    def print_decision_tree(self, node=None):
        if node is None:
            print('ROOT\n-----')
            node = self.head
        node.print_node()
        if not node.is_leaf:
            print('left son:')
            self.print_decision_tree(node.left_son)
            print('right son:')
            self.print_decision_tree(node.right_son)

    def dynamic_partition(self, column, node):
        # print(f' ID3::dynamic_partition - start')
        # start = time.time()

        sorted_ascending_f_val = sorted(node.example_set, key=itemgetter(column))
        best_threshold = -1
        best_ig = 0
        set_size = len(sorted_ascending_f_val)
        best_ig_assigned = False
        best_h_e_1 = best_h_e_2 = 0
        best_i = 1

        for i in range(self.min_node_size - 1, set_size - self.min_node_size):
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
        if not best_ig_assigned:
            node.set_as_a_leaf()
            return False, 0, 0, None, None, 0, 0
        part_left_set = sorted_ascending_f_val[:best_i]
        part_right_set = sorted_ascending_f_val[best_i:]

        # end = time.time()
        # print(f'dynamic_partition took {end - start} second')
        # print('ID3::dynamic_partition - end')
        return True, best_ig, best_threshold, part_left_set, part_right_set, \
               best_h_e_1, best_h_e_2

    def classifier(self, subject):
        if not self.is_tree_initialize:
            self.create_classifier_tree()
        node = self.head
        while not node.is_leaf:
            subj_f_val = subject[node.separating_f]
            if subj_f_val >= node.threshold:
                node = node.right_son
            else:
                node = node.left_son
        assert node.is_leaf
        return node.leaf_diagnosis

    def analyze_node_set(self, node_set):
        m_b_count = {'B': 0, 'M': 0}
        for example in node_set:
            m_b_count[example[0]] += 1
        default = 'M' if m_b_count['M'] > m_b_count['M'] else 'B'
        is_leaf = (m_b_count['B'] == 0 or m_b_count['M'] == 0) or (len(node_set) < self.min_node_size)
        classification = None
        if is_leaf:
            if m_b_count['M'] == 0:
                classification = 'B'
            elif m_b_count['B'] == 0:
                classification = 'M'
            else:
                assert False
        return is_leaf, classification, default


if __name__ == '__main__':
    id3 = ID3('/home/guy-pc/PycharmProjects/ID3/train.csv', min_node_set_size=5)
    id3.create_classifier_tree()
    id3.print_decision_tree()
    test = pan.read_csv('/home/guy-pc/PycharmProjects/ID3/test.csv')
    correct_answers = 0
    subjects_size = test['diagnosis'].size
    for subj in test.iloc:
        diagnosis = id3.classifier(subj)
        real_diagnosis = subj['diagnosis']
        if diagnosis == real_diagnosis:
            correct_answers += 1
    print(f'{correct_answers / subjects_size}')
