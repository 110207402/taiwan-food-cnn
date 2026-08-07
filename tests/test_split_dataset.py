import tempfile
import unittest
from pathlib import Path

from scripts.split_dataset import split_dataset


class SplitDatasetTest(unittest.TestCase):
    def test_deterministic_stratified_split(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "raw"
            for class_name in ("bawan", "bubble_tea"):
                class_dir = source / class_name
                class_dir.mkdir(parents=True)
                for index in range(10):
                    (class_dir / f"{index}.jpg").write_bytes(b"synthetic-test-file")

            first = root / "first"
            second = root / "second"
            totals = split_dataset(source, first, seed=42)
            split_dataset(source, second, seed=42)

            self.assertEqual(totals, {"train": 16, "val": 2, "test": 2})
            self.assertEqual(
                (first / "dataset_split.csv").read_text(),
                (second / "dataset_split.csv").read_text(),
            )


if __name__ == "__main__":
    unittest.main()
