"""Dataset classes for exoskeleton data."""

from pathlib import Path
from typing import List

import pandas as pd
import torch
from torch.utils.data import Dataset

class ExoDataset(Dataset):
    def __init__(self,
                    data_path: str | Path,
                    input_names: List[str],
                    label_names: List[str],
                    split: str = "train",
                    device: torch.device = torch.device("cpu")):
        self.data_path = Path(data_path)
        self.input_names = input_names
        self.label_names = label_names
        self.device = device
    
    def _load_all(self) -> list[tuple[torch.Tensor,torch.Tensor]]: # TODO: dependent on file structure
        data = []

        INPUT_CSV = 'Exo.csv'
        LABEL_CSV = 'Joint_Moments_Filt.csv'


        for input_file_path in self.data_path.glob(f'**/{INPUT_CSV}'):
            label_file_path = input_file_path.parent / LABEL_CSV

            data.append((
                self._load_by_headers(input_file_path,self.input_names),
                self._load_by_headers(label_file_path,self.label_names)
            ))
        
        return data
    
    def _load_by_headers(self, file_path: str | Path, headers: List[str]) -> torch.Tensor: # size: [1, # of headers, # of timestamps in csv]
        # load as DataFrame
        df = pd.read_csv(file_path)

        # convert to input and label tensors
        label_data = torch.tensor(df[headers].values, device = self.device)
        label_data = label_data.transpose(0, 1).unsqueeze(0).float() 
        print(label_data.size())
        
        return label_data
            


if __name__ == "__main__":
    input_names = [ "hip_angle_r", "hip_angle_r_velocity_filt", "hip_angle_l", "hip_angle_l_velocity_filt", 
                    "knee_angle_r", "knee_angle_r_velocity_filt", "knee_angle_l", "knee_angle_l_velocity_filt"]

    # corresponding model label names in dataset
    label_names = ["hip_flexion_l_moment", "knee_angle_l_moment", "hip_flexion_r_moment", "knee_angle_r_moment"]

    data_path = "..."
    
    testds = ExoDataset(data_path,input_names,label_names)
    testds._load_all()
