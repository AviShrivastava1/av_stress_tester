from pathlib import Path
import sys


class ShardLoader:
    """
    Opens a .tfrecord shard and iterates over raw serialized scenario bytes.
    Each record is one scenario, still serialized — the parser will decode it.
    """

    def __init__(self, shard_path: str):
        self.shard_path = Path(shard_path)

        if not self.shard_path.exists():
            raise FileNotFoundError(f"Shard not found: {self.shard_path}")

    def __iter__(self):
        """
        Iterate over every record in the shard.
        Yields raw bytes — one blob per scenario.
        """
        dataset = tf.data.TFRecordDataset(str(self.shard_path))
        for raw_bytes in dataset:
            yield raw_bytes.numpy()

    def __len__(self):
        """
        Count total number of scenarios in this shard.
        Useful for progress tracking in the batch scorer later.
        """
        dataset = tf.data.TFRecordDataset(str(self.shard_path))
        return sum(1 for _ in dataset)


if __name__ == "__main__":
    print("script started")

    shard_path = sys.argv[1]
    print(f"shard path received: {shard_path}")

    loader = ShardLoader(shard_path)
    print(f"loader created successfully")

    print(f"Total scenarios: {len(loader)}")

    for i, raw_bytes in enumerate(loader):
        print(f"Scenario {i}: {len(raw_bytes)} bytes")
        if i >= 4:
            print("...")
            break

    print("done")