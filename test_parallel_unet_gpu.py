import torch
import timeit
import cProfile
from tryondiffusion import get_unet_by_name

IMAGE_SIZE = (128, 128)  # (height, width), change to (256, 256) for "sr" unet)
BATCH_SIZE = 2
MODEL_NAME = "base"  # "base" or "sr"
MODEL_KWARGS = {}
LOWRES_IMAGE = MODEL_NAME == "sr"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
def main():
    unet = get_unet_by_name(MODEL_NAME, **MODEL_KWARGS).to(device)
    print(f"Model: {unet.__class__.__name__}")
    print(f"Model parameters: {sum(p.numel() for p in unet.parameters()) / 1e9:.2f}B")

    noisy_images = torch.randn(BATCH_SIZE, 3, *IMAGE_SIZE, device=device)
    lowres_cond_images = torch.randn(BATCH_SIZE, 3, *IMAGE_SIZE, device=device) if LOWRES_IMAGE else None

    ca_images = torch.randn(BATCH_SIZE, 3, *IMAGE_SIZE, device=device)
    garment_images = torch.randn(BATCH_SIZE, 3, *IMAGE_SIZE, device=device)
    person_poses = torch.randn(BATCH_SIZE, 18, 2, device=device)
    garment_poses = torch.randn(BATCH_SIZE, 18, 2, device=device)

    time = torch.randn(BATCH_SIZE, device=device)
    lowres_noise_times = torch.randn(BATCH_SIZE, device=device) if LOWRES_IMAGE else None

    ca_noise_times = torch.randn(BATCH_SIZE, device=device)
    garment_noise_times = torch.randn(BATCH_SIZE, device=device)

    print(f"noisy_images shape: {noisy_images.shape}")

    # Sync before timing to avoid startup overhead
    if device.type == 'cuda':
        torch.cuda.synchronize()

    output = unet(
        noisy_images=noisy_images,
        time=time,
        lowres_cond_img=lowres_cond_images,
        lowres_noise_times=lowres_noise_times,
        ca_images=ca_images,
        ca_noise_times=ca_noise_times,
        garment_images=garment_images,
        garment_noise_times=garment_noise_times,
        person_poses=person_poses,
        garment_poses=garment_poses,
    )

    # Sync after forward pass to get accurate timing
    if device.type == 'cuda':
        torch.cuda.synchronize()

    print(f"output shape: {output.shape}")


if __name__ == "__main__":
    # Measure execution time including CUDA sync for accurate timing on GPU
    def timed_main():
        if device.type == 'cuda':
            torch.cuda.synchronize()
        main()
        if device.type == 'cuda':
            torch.cuda.synchronize()

    execution_time = timeit.timeit(timed_main, number=1)
    print(f"Execution time: {execution_time:.4f} seconds")

    print("\nProfiling results:")
    cProfile.run('main()')
