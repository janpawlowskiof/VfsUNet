def get_dataset(config):
    dataset_name = config["name"]

    if dataset_name == "CompressionDataset":
        from src.dataset.compression_dataset import CompressionDataset
        return CompressionDataset.from_config(config)
    elif dataset_name == "Empty":
        return None
    raise NotImplementedError()
