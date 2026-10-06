def get_qm9(root="./tmp/qm9"):
    # get the full qm9 dataset
    from torch_geometric.datasets import QM9

    return QM9(root=root)


def get_mp20(split="all"):
    # get the full mp20 dataset
    from StructureCloud.Datasets import MP20_dataset

    return MP20_dataset(split=split, preprocess=True)
