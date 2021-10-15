import decision_tree
import numpy as np


class Forest:
    def __init__(self, trees):
        self.trees = trees


def bagging(data, target, B=100):
    randoms = np.random.randint(0, len(data), size=B)
    return data[randoms], target[randoms]


def create_random_forest(data, target, n_trees=100, n_features=0, B=100):
    if not n_features:
        n_features = int(np.sqrt(data.shape[1]))
    trees = []
    for _ in range(n_trees):
        random_features = np.random.randint(0, data.shape[1], size=n_features)
        bagged_data, bagged_target = bagging(data, target, B)
        trees.append(decision_tree.Tree(bagged_data, bagged_target, random_features))
    return Forest(trees)


def evaluate_random_forest(forest, data, target):
    pred_y = []
    for x, y in zip(data, target):
        preds_y_for_x = []
        for tree in forest.trees:
            preds_y_for_x.append(tree.predict(x))
        pred_y.append(np.argmax(np.bincount(np.array(preds_y_for_x))))
    pred_y = np.array(pred_y)
    return np.array(pred_y), decision_tree.accuracy(pred_y, target)


if __name__ == '__main__':
    tr_data, tr_target, val_data, val_target, te_data, te_target = decision_tree.create_dataset()
    for _ in range(10):
        forest = create_random_forest(tr_data, tr_target, n_trees=20, B=tr_data.shape[0])
        pred, score = evaluate_random_forest(forest, te_data, te_target)
        print(score)