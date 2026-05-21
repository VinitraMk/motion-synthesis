import random
from pathlib import Path

def make_debug_splits(
    train_split_path,
    val_split_path,
    out_dir,
    train_debug_n=64,
    val_debug_n=32,
    micro_train_n=6,
    micro_val_n=3,
    seed=1234,
):
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(train_split_path, "r") as f:
        train_ids = [line.strip() for line in f if line.strip()]

    with open(val_split_path, "r") as f:
        val_ids = [line.strip() for line in f if line.strip()]

    rng.shuffle(train_ids)
    rng.shuffle(val_ids)

    train_debug = train_ids[:train_debug_n]
    val_debug = val_ids[:val_debug_n]
    train_micro = train_ids[:micro_train_n]
    val_micro = val_ids[:micro_val_n]

    (out_dir / "train_debug.txt").write_text("\n".join(train_debug) + "\n")
    (out_dir / "val_debug.txt").write_text("\n".join(val_debug) + "\n")
    (out_dir / "train_micro.txt").write_text("\n".join(train_micro) + "\n")
    (out_dir / "val_micro.txt").write_text("\n".join(val_micro) + "\n")

    print("train_debug:", len(train_debug))
    print("val_debug:", len(val_debug))
    print("train_micro:", len(train_micro))
    print('val_mciro:', len(val_micro))


if __name__ == "__main__":
    make_debug_splits("./data/HumanML3D/train.txt", "./data/HumanML3D/val.txt", "./data/HumanML3D")