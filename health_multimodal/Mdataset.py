from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference
from health_multimodal.image.data.io import load_image

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from pathlib import Path
from typing import Callable, Tuple

TypeShape2D = Tuple[int, int]

TRANSFORM_RESIZE = 448
CENTER_CROP_SIZE = 448
map ={'Pleural_Effusion_Status':-6, 'Edema_Status':-5, 'Consolidation_Status':-4, 'Pneumonia_Status':-3, 'Pneumothorax_Status':-2}
class MimicDataset(Dataset):
    def __init__(self, csv_path, findings, is_test=False):
        self.findings=findings
        self.label_key =f"{self.findings}_Status"
        self.label_index= map[self.label_key]
        csv = pd.read_csv(csv_path)  # CSV 파일 로드
        criteria =  "test" if is_test else "train" 
        self.data = csv[(csv[self.label_key]>1) & (csv["mm_split"]==criteria)].values
        # self.features = self.data.iloc[:, :-1].values  # 마지막 열 제외 (입력)
        # self.labels = self.data.iloc[:, -1].values  # 마지막 열 (정답)
        self.transform = create_chest_xray_transform_for_inference(TRANSFORM_RESIZE, CENTER_CROP_SIZE)  # 변환 정의

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_tensor, _ = self.load_and_transform_input_image(self.data[idx][0], self.transform)
        y_tensor, _ = self.load_and_transform_input_image(self.data[idx][1], self.transform)
        label = torch.tensor(self.data[idx][self.label_index] -2)
        return x_tensor, y_tensor, label

    def load_and_transform_input_image(self, image_path: str, transform: Callable) -> Tuple[torch.Tensor, TypeShape2D]:
        image = load_image(Path(f"/data/public/mimic/preprocessed/{image_path}.png"))
        transformed_image = transform(image)
        return transformed_image, image.size
