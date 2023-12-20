

import numpy as np


def CUB_iid(dataset,k):
	num_items = int(len(dataset) / k)

	dict_users, all_idxs = {}, [i for i in range(len(dataset))]

	for i in range(k):
		dict_users[i] = set(np.random.choice(all_idxs,num_items,replace=False))#select #num_items indexs from all indexs
		# all_idxs.pop(dict_users[i])
		all_idxs = list(set(all_idxs) - dict_users[i])#remove idx selected

	return dict_users

def CUB_non_iid(dataset, k):
    """
    Sample non-I.I.D client data from dataset
    -> Different clients can hold vastly different amounts of data
    :param dataset:
    :param num_users:
    :return:
    """
    num_dataset = len(dataset)
    idx = np.arange(num_dataset)
    dict_users = {i: list() for i in range(k)}
    
    min_num = 100
    max_num = 700

    random_num_size = np.random.randint(min_num, max_num+1, size=k)
    print(f"Total number of datasets owned by clients : {sum(random_num_size)}")

    # total dataset should be larger or equal to sum of splitted dataset.
    assert num_dataset >= sum(random_num_size)

    # divide and assign
    for i, rand_num in enumerate(random_num_size):

        rand_set = set(np.random.choice(idx, rand_num, replace=False))
        idx = list(set(idx) - rand_set)
        dict_users[i] = rand_set

    return dict_users
