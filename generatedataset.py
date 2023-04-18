from datasets import ImageNetDataset

ImageNetDataset.make_dataset_index(
    train_data_path = "./data/train",
    val_data_path = "./data/eval",
    test_data_path= "./data/test",
    index_path = "./data"
)