"""
Uploads a trained checkpoint to a Hugging Face Hub model repo.

One-time setup (needs a WRITE-scoped token from
https://huggingface.co/settings/tokens):
    huggingface-cli login

Usage:
    python upload_checkpoint.py \\
        --checkpoint ./checkpoints_kaggle_hf_zalando/checkpoint.47900.pt \\
        --repo-id <your-username>/digital-tryon-v2

Checkpoint files here run tens of GB; huggingface_hub's upload_file()
automatically tracks anything over 10MB with Git LFS and uploads it in
chunks, so this handles large files correctly without extra flags -
expect the upload itself to take a while depending on your upstream
bandwidth.
"""
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Local checkpoint .pt file to upload")
    parser.add_argument(
        "--repo-id", required=True,
        help="Target HF Hub model repo, e.g. your-username/digital-tryon-v2 (created automatically if it doesn't exist)"
    )
    parser.add_argument(
        "--path-in-repo", default=None,
        help="Filename inside the repo (default: same as the local filename, e.g. checkpoint.47900.pt)"
    )
    parser.add_argument("--private", action="store_true", help="Create the repo as private (default: public)")
    parser.add_argument("--commit-message", default=None, help="Commit message for this upload")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    from huggingface_hub import HfApi, create_repo

    path_in_repo = args.path_in_repo or os.path.basename(args.checkpoint)
    size_gb = os.path.getsize(args.checkpoint) / (1024 ** 3)
    print(f"Uploading {args.checkpoint} ({size_gb:.1f} GB) to {args.repo_id}/{path_in_repo} ...")
    print("This can take a while for large files - progress is printed by huggingface_hub below.")

    create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    api = HfApi()
    api.upload_file(
        path_or_fileobj=args.checkpoint,
        path_in_repo=path_in_repo,
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.commit_message or f"Upload {path_in_repo}",
    )

    print(f"Done: https://huggingface.co/{args.repo_id}/blob/main/{path_in_repo}")


if __name__ == "__main__":
    main()
