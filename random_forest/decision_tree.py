# # import numpy as np
# # import sys
# #
# #
# # class Node():
# #     def __init__(self, X, Y, parent):
# #         self.X = X
# #         self.Y = Y
# #         self.left = None
# #         self.right = None
# #         self.parent = parent
# #         self.feature = -1
# #         self.threshold = -1
# #         self.entropy = -1
# #
# #     def is_leaf(self):
# #         return not (self.left or self.right)
# #
# #     def equals_parent(self):
# #         if not self.parent:
# #             return False
# #         if self.X.shape == self.parent.X.shape:
# #             return np.all(np.equal(self.X, self.parent.X))
# #         return False
# #
# #     def is_empty(self):
# #         return self.X.size == 0 or self.equals_parent()
# #
# #     def split(self, features):
# #         best_feature, best_threshold, split_left_indices, split_right_indices, min_entropy = Node.find_split(self.X, self.Y, features)
# #         self.feature = best_feature
# #         self.threshold = best_threshold
# #         self.entropy = min_entropy
# #         if self.is_empty():
# #             self.parent.left = self.parent.right = None
# #             self.left = self.right = None
# #             return
# #         if self.entropy != 0:
# #             self.left = Node(self.X[split_left_indices], self.Y[split_left_indices], parent=self)
# #             self.right = Node(self.X[split_right_indices], self.Y[split_right_indices], parent=self)
# #
# #     @staticmethod
# #     def find_split(data, target, features=None):
# #         if features is None:
# #             features = range(data.shape[1])
# #         min_entropy = sys.maxsize
# #         split_left_indices, split_right_indices = 0, 0
# #         best_feature, best_threshold = 0, 0
# #         for feature in features:
# #             unique_xis = np.unique(data[:, feature])
# #             col = data[:, feature]
# #             for threshold in unique_xis:
# #                 cond_entropy, gt_args, lt_args = Node.find_best_threshold(col, target, threshold)
# #                 if cond_entropy < min_entropy:
# #                     best_feature = feature
# #                     best_threshold = threshold
# #                     split_left_indices, split_right_indices = lt_args, gt_args
# #                     min_entropy = cond_entropy
# #         return best_feature, best_threshold, split_left_indices, split_right_indices, min_entropy
# #
# #     @staticmethod
# #     def find_best_threshold(feature, target, threshold):
# #         lt = np.where(feature <= threshold, 1, 0)
# #         p_lt = np.sum(lt) / len(lt)
# #         p_gt = 1 - p_lt
# #         lt_args = np.argwhere(feature <= threshold).flatten()
# #         gt_args = np.argwhere(feature > threshold).flatten()
# #         y_for_lt = np.bincount(target[lt_args])
# #         y_for_lt = y_for_lt[y_for_lt > 0]
# #         y_for_gt = np.bincount(target[gt_args])
# #         y_for_gt = y_for_gt[y_for_gt > 0]
# #         p_y_lt_threshold = y_for_lt / y_for_lt.sum()
# #         p_y_gt_threshold = y_for_gt / y_for_gt.sum()
# #         cond_entropy = p_lt * -(p_y_lt_threshold * np.log2(p_y_lt_threshold)).sum() + \
# #                        p_gt * -(p_y_gt_threshold * np.log2(p_y_gt_threshold)).sum()
# #         return cond_entropy, gt_args, lt_args
# #
# #     def test(self, X):
# #         if not (self.left or self.right):
# #             return self
# #         elif X[self.feature] <= self.threshold:
# #             return self.left
# #         else:
# #             return self.right
# #
# #     def get_target(self):
# #         return np.argmax(np.bincount(self.Y, minlength=11))
# #
# #
# # def accuracy(pred, true):
# #     return np.sum(pred == true) / pred.size
# #
# #
# # class Tree:
# #     def __init__(self, data, target, features):
# #         self.root = Node(data, target, None)
# #         self.leaves = []
# #         self.features = features
# #         self._create_tree(self.root)
# #
# #     def _create_tree(self, node):
# #         queue = [node]
# #         while queue:
# #             current = queue.pop(0)
# #             current.split(self.features)
# #             if current:
# #                 if current.left and (not current.left.is_empty()):
# #                     queue.append(current.left)
# #                 if current.right and (not current.right.is_empty()):
# #                     queue.append(current.right)
# #
# #     def evaluate_tree(self, X, Y_true):
# #         pred = np.array([self.predict(x) for x in X])
# #         score = accuracy(pred, Y_true)
# #         return pred, score
# #
# #     def predict(self, x):
# #         return Tree._predict(self.root, x)
# #
# #     @staticmethod
# #     def _predict(node, x):
# #         successor = node.test(x)
# #         if successor == node:
# #             return node.get_target()
# #         return Tree._predict(successor, x)
# #
# #     def _find_leaves(self, node):
# #         if not node:
# #             return
# #         if node.is_leaf():
# #             self.leaves.append(node)
# #         self._find_leaves(node.left)
# #         self._find_leaves(node.right)
# #
# #     def prune_tree(self, val_data, val_target, epsilon=0.):
# #         self._find_leaves(self.root)
# #         prunnable = {leaf.parent for leaf in self.leaves}
# #         checked = set()
# #         while prunnable:
# #             _, org_score = self.evaluate_tree(val_data, val_target)
# #             to_prune = prunnable.pop()
# #             checked.add(to_prune)
# #             to_prune_left, to_prune_right = to_prune.left, to_prune.right
# #             to_prune.left = to_prune.right = None
# #             _, pruned_score = self.evaluate_tree(val_data, val_target)
# #             if pruned_score <= (org_score - epsilon):
# #                 to_prune.left, to_prune.right = to_prune_left, to_prune_right
# #             else:
# #                 self._find_leaves(self.root)
# #                 prunnable = {leaf.parent for leaf in self.leaves if leaf.parent} - checked
# #
# #
# # def create_dataset(tr_ratio=5, val_ratio=1, te_ratio=1):
# #     dataset = np.genfromtxt('winequality-white.csv', delimiter=';', dtype=str)
# #     dataset = dataset[1:].astype(np.float)
# #     np.random.shuffle(dataset)
# #     tr_len = int(tr_ratio / (tr_ratio + val_ratio + te_ratio) * len(dataset))
# #     val_len = int((tr_ratio + val_ratio) / (tr_ratio + val_ratio + te_ratio) * len(dataset))
# #     tr_set, val_set, te_set = np.split(dataset, [tr_len, val_len])
# #     return tr_set[:, :-1], tr_set[:, -1].astype(int),\
# #            val_set[:, :-1], val_set[:, -1].astype(int),\
# #            te_set[:, :-1], te_set[:, -1].astype(int)
# #
# # if __name__ == '__main__':
# #     for _ in range(10):
# #         tr_data, tr_target, val_data, val_target, te_data, te_target = create_dataset()
# #         tree = Tree(tr_data, tr_target, features=None)
# #
# #         _, score = tree.evaluate_tree(te_data, te_target)
# #         print(f'Before pruning: {score}')
# #         tree.prune_tree(val_data, val_target, epsilon=0.002)
# #         _, pruned_score = tree.evaluate_tree(te_data, te_target)
# #         print(f'After pruning: {pruned_score}')
#
# import numpy as np
#
# arr = np.array([[2,3,5], [4,9,25], [8,27,125]])
#
# print(arr[:,1:3])

import numpy as np
# from src.sel_eval import Dataset, Split
#
# def cv_splitter(dataset, n_splits, seed):
#     rng = np.random.RandomState(seed)
#     np.random.seed(seed)
#     np.random.shuffle(dataset)
#     test_size = dataset.shape[0] / n_splits
#     # UZUPEŁNIĆ

seed = 43
dataset = [0,1,2]
print(np.mean(dataset))