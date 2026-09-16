"""
Deletes old checkpoint.<N>.pt files in a checkpoint directory, keeping only
the N highest-step ones. Safe by default: without --apply, it only prints
what it *would* delete (and how much disk space that frees) - nothing is
actually removed until you pass --apply.

Usage:
    python prune_checkpoints.py --dir ./checkpoints_base_kaggle --keep 2            # dry run (preview only)
    python prune_checkpoints.py --dir ./checkpoints_base_kaggle --keep 2 --apply    # actually delete
"""
import argparse
import glob
import os


def step_of(path: str) -> int:
    try:
        return int(os.path.basename(path).split(".")[-2])
    except (IndexError, ValueError):
        return -1


def human_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}PB"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True, help="Checkpoint directory (e.g. ./checkpoints_base_kaggle)")
    parser.add_argument("--keep", type=int, default=2, help="Number of highest-step checkpoints to keep")
    parser.add_argument("--apply", action="store_true", help="Actually delete files (default: dry run / preview only)")
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        raise FileNotFoundError(f"Not a directory: {args.dir}")

    checkpoints = glob.glob(os.path.join(args.dir, "checkpoint.*.pt"))
    if not checkpoints:
        print(f"No checkpoint.<N>.pt files found in {args.dir}")
        return

    checkpoints.sort(key=step_of, reverse=True)
    keep, remove = checkpoints[: args.keep], checkpoints[args.keep :]

    print(f"Found {len(checkpoints)} checkpoint(s) in {args.dir}\n")

    print(f"KEEP ({len(keep)}):")
    for p in keep:
        print(f"  {os.path.basename(p)}  ({human_size(os.path.getsize(p))})")

    if not remove:
        print("\nNothing to delete - already at or below --keep count.")
        return

    total_freed = sum(os.path.getsize(p) for p in remove)
    print(f"\n{'DELETE' if args.apply else 'WOULD DELETE'} ({len(remove)}, frees {human_size(total_freed)}):")
    for p in remove:
        print(f"  {os.path.basename(p)}  ({human_size(os.path.getsize(p))})")

    if not args.apply:
        print("\nDry run only - nothing was deleted. Re-run with --apply to actually delete the files above.")
        return

    for p in remove:
        os.remove(p)
    print(f"\nDeleted {len(remove)} file(s), freed {human_size(total_freed)}.")


if __name__ == "__main__":
    main()
