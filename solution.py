from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimized TV-regularized image deconvolution")
    parser.add_argument("input_image", type=str)
    parser.add_argument("kernel", type=str)
    parser.add_argument("output_image", type=str)
    parser.add_argument("noise_level", type=float)
    return parser


def load_grayscale_bmp(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image.astype(np.float32)


def save_grayscale_bmp(path: str | Path, image: np.ndarray) -> None:
    image_u8 = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(path), image_u8):
        raise OSError(f"Could not write image: {path}")


def normalize_kernel(kernel: np.ndarray) -> np.ndarray:
    kernel = np.asarray(kernel, dtype=np.float32)
    total = float(kernel.sum())
    if total <= 0.0:
        raise ValueError("Kernel sum must be positive.")
    return kernel / total


def convolve_reflect(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return cv2.filter2D(image, cv2.CV_32F, kernel[::-1, ::-1], borderType=cv2.BORDER_REFLECT)


def gradient_forward(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grad_x = np.empty_like(image, dtype=np.float32)
    grad_y = np.empty_like(image, dtype=np.float32)
    grad_x[:, :-1] = image[:, 1:] - image[:, :-1]
    grad_x[:, -1] = 0.0
    grad_y[:-1, :] = image[1:, :] - image[:-1, :]
    grad_y[-1, :] = 0.0
    return grad_x, grad_y


def divergence(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    div = np.empty_like(px, dtype=np.float32)
    div[:, 0] = px[:, 0]
    div[:, 1:-1] = px[:, 1:-1] - px[:, :-2]
    div[:, -1] = -px[:, -2]
    div[0, :] += py[0, :]
    div[1:-1, :] += py[1:-1, :] - py[:-2, :]
    div[-1, :] += -py[-2, :]
    return div


def choose_parameters(noise_level: float) -> tuple[float, int]:
    lambda_tv = 0.111471 * np.power(float(noise_level) + 1.0, 1.419654) - 0.253609
    if noise_level <= 2.0:
        lambda_tv *= 0.85
    lambda_tv = float(np.clip(lambda_tv, 0.01, 8.0))
    return lambda_tv, 100


def deconvolve_tv(blurred: np.ndarray, kernel: np.ndarray, noise_level: float) -> np.ndarray:
    kernel = normalize_kernel(kernel)
    kernel_flipped = kernel[::-1, ::-1]

    lambda_tv, iterations = choose_parameters(noise_level)
    tau = 1.2
    sigma = 0.05
    theta = 1.0

    x = blurred.copy()
    x_bar = x.copy()
    px = np.zeros_like(blurred, dtype=np.float32)
    py = np.zeros_like(blurred, dtype=np.float32)

    for _ in range(iterations):
        grad_x, grad_y = gradient_forward(x_bar)
        px += sigma * grad_x
        py += sigma * grad_y

        norms = np.maximum(1.0, np.sqrt(px * px + py * py) / lambda_tv)
        px /= norms
        py /= norms

        blurred_estimate = convolve_reflect(x, kernel)
        data_grad = convolve_reflect(blurred_estimate - blurred, kernel_flipped)
        x_new = np.clip(x - tau * (data_grad - divergence(px, py)), 0.0, 255.0)
        x_bar = x_new + theta * (x_new - x)
        x = x_new

    return x


def main() -> None:
    args = build_parser().parse_args()

    blurred = load_grayscale_bmp(args.input_image)
    kernel = load_grayscale_bmp(args.kernel)
    restored = deconvolve_tv(blurred=blurred, kernel=kernel, noise_level=args.noise_level)
    save_grayscale_bmp(args.output_image, restored)


if __name__ == "__main__":
    main()
