import struct
from pathlib import Path


class ShardLoader:
    """
    Opens a .tfrecord shard and iterates over raw serialized scenario bytes.
    Reads the tfrecord binary format directly — no TensorFlow needed.

    TFRecord format per record:
        uint64  length of data
        uint32  masked crc32 of length
        bytes   data (the serialized protobuf)
        uint32  masked crc32 of data
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
        with open(self.shard_path, 'rb') as f:
            while True:
                header = f.read(8)
                if len(header) < 8:
                    break

                length = struct.unpack('<Q', header)[0]
                f.read(4)
                data = f.read(length)
                f.read(4)

                yield data