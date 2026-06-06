import torch.utils.data

import torch
import os
import pandas as pd
import glob
import cv2
import albumentations as A
import logging
import numpy.typing as npt
import numpy as np
import random

IMAGE_SIZE = (24, 24)

def train_tensor_to_255_numpy(tensor: torch.Tensor) -> npt.NDArray[np.uint8]:
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(3, 1, 1)
    image = tensor * std + mean

    image = image.cpu().numpy().transpose(1, 2, 0)  # CxHxW to HxWxC

    image = (image * 255).clip(0, 255).astype(np.uint8)

    assert image.shape[2] == 3, f"{image.shape=}"

    return image


class CarDataset(torch.utils.data.Dataset):
    def __init__(self, ride_dir, *, future_horizont, past_horizont, transform=None, device="cpu", ):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.past_horizont = past_horizont
        self.forward_horizont = future_horizont

        if transform is None:
            self.transform = A.Compose(
                [
                    A.Resize(*IMAGE_SIZE),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    A.pytorch.ToTensorV2(),
                ]
            )

        assert os.path.isdir(ride_dir), f"Directory {ride_dir} does not exist."
        self.logger.debug(f"Found directory {ride_dir} for dataset.")

        csv_file_names = glob.glob(os.path.join(ride_dir, "*.csv"))

        assert len(csv_file_names) > 0, f"No CSV files found in {ride_dir}."
        self.logger.debug(f"Found {len(csv_file_names)} CSV files in {ride_dir}.")

        csv_dfs = []
        self.image_dir_names = []
        self.image_sequences = []
        self.signals_sequences = []

        for csv_file_name in csv_file_names:
            run_id, _ = os.path.splitext(os.path.basename(csv_file_name))

            df = pd.read_csv(csv_file_name)
            csv_dfs.append(df)
            image_dir_name = os.path.join(ride_dir, run_id)
            self.image_dir_names.append(image_dir_name)
            images_curr = []
            signals_curr = []

            for row in df.itertuples():

                assert len(row) >= 4  # first is index

                image_path = os.path.join(image_dir_name, str(row[1]).zfill(4) + ".jpg")
                if not os.path.isfile(image_path):
                    raise FileNotFoundError(
                        f"Image file {image_path} does not exist for csv {csv_file_name}."
                    )
                image = self.load_image(image_path)
                image = self.transform(image=image)["image"]
                images_curr.append(image)

                forward_signal = row[2]
                left_signal = row[3]
                signals_curr.append([forward_signal, left_signal])
            images = torch.stack(images_curr, dim=0).to(device)
            # pad past from the left and future horizons from the right with zeros
            
            images_padded = torch.zeros(
                (len(images_curr) + self.past_horizont + self.forward_horizont, *images.shape[1:]),
                dtype=images.dtype,
                device=device,
            )
            images_padded[self.past_horizont : self.past_horizont + len(images_curr)] = images
            self.image_sequences.append(images_padded)

            signals = torch.tensor(signals_curr, dtype=torch.float32, device=device)
            signals_padded = torch.zeros(
                (len(signals_curr) + self.past_horizont + self.forward_horizont, signals.shape[1]),
                dtype=signals.dtype,
                device=device,
            )
            signals_padded[self.past_horizont : self.past_horizont + len(signals_curr)] = signals

            assert signals.ndim == 2
            assert signals_padded.shape[0] == 2

            self.signals_sequences.append(
                signals_padded
            )


            assert (
                self.image_sequences[-1].shape[0]
                == self.signals_sequences[-1].shape[0]
            ), (
                f"Number of images and signals do not match in {csv_file_name} after stacking."
            )

            self.logger.debug(
                f"Loaded {len(images_curr)} samples from {csv_file_name}."
            )

    def load_image(self, image_path):
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file {image_path} does not exist.")

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image {image_path}.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __len__(self):
        return len(self.signals_sequences)

    def __getitem__(self, idx): 
        if idx < 0 or idx >= len(self.signals_sequences): raise IndexError(
                f"Index {idx} is out of bounds for dataset of size {len(self.signals_sequences)}."
            )

        images, signals = self.image_sequences[idx], self.signals_sequences[idx]

        random_frame_id = random.randint(self.past_horizont, images.shape[0] - self.forward_horizont - 1)

        past_images = images[random_frame_id - self.past_horizont : random_frame_id]
        future_signals = signals[random_frame_id : random_frame_id + self.forward_horizont]

        # assert not (gur[0] == 0.0).all(), f"First past image is all zeros for index {idx} and frame id {random_frame_id}."

        return past_images, future_signals

        



def display_dataset(dataset: CarDataset, fps: int):
    import raylib as rl
    import pyray as pyrl

    WINDOW_WIDTH = 448
    WINDOW_HEIGHT = 448

    rl.InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, b"Car Dataset Visualization")
    rl.SetTargetFPS(fps)

    ride_index = 0
    dummy_img = np.zeros((WINDOW_WIDTH, WINDOW_HEIGHT, 3), dtype=np.uint8)
    img_struct = pyrl.Image()
    img_struct.width = WINDOW_WIDTH
    img_struct.height = WINDOW_HEIGHT
    img_struct.format = rl.PIXELFORMAT_UNCOMPRESSED_R8G8B8
    img_struct.mipmaps = 1
    img_struct.data = rl.ffi.cast("void *", dummy_img.ctypes.data)

    texture = rl.LoadTextureFromImage(img_struct)

    current_datapoints = None
    current_frame_index = 0
    while not rl.WindowShouldClose():
        print(
            f"Current ride index: {ride_index}, current frame index: {current_frame_index}",
            flush=True,
        )

        if rl.IsKeyPressed(rl.KEY_RIGHT):
            ride_index = (ride_index + 1) % len(dataset)
            current_datapoints = None
            current_frame_index = 0

        elif rl.IsKeyPressed(rl.KEY_LEFT):
            ride_index = (ride_index - 1) % len(dataset)
            current_datapoints = None
            current_frame_index = 0

        if current_datapoints is None:
            ride_images, ride_signals = dataset[ride_index]
            assert ride_images.shape[0] == ride_signals.shape[0], (
                f"Number of images and signals do not match for current ride (index {ride_index})."
            )
            current_datapoints = []
            for i in range(ride_images.shape[0]):
                image = train_tensor_to_255_numpy(ride_images[i])
                image = cv2.resize(image, (WINDOW_WIDTH, WINDOW_HEIGHT))
                forward_signal, left_signal = ride_signals[i].cpu().numpy()
                current_datapoints.append((np.ascontiguousarray(image), (forward_signal, left_signal)))

        current_frame, current_signal = current_datapoints[current_frame_index]

        rl.UpdateTexture(texture, rl.ffi.cast("void *", current_frame.ctypes.data))

        rl.BeginDrawing()
        rl.ClearBackground(rl.RAYWHITE)
        rl.DrawTexture(texture, 0, 0, (255, 255, 255, 255))
        rl.DrawText(
            f"Forward: {current_signal[0]:.2f}, Left: {current_signal[1]:.2f}".encode("utf-8"),
            10,
            WINDOW_HEIGHT - 30,
            20,
            rl.BLACK,
        )
        rl.DrawFPS(5, 5)
        rl.EndDrawing()
        current_frame_index = (current_frame_index + 1) % len(current_datapoints)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = CarDataset("./dataset/", device=device, future_horizont=5, past_horizont=3)
    # print(f"Dataset length: {len(dataset)}")
    # display_dataset(dataset, 24)


