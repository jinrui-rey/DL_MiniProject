# DL_MiniProject
MiniProject of Deep Learning 2024Spring
## Requirments

### Python
- Python 3.6+
- PyTorch 1.6.0+
### Libary
- torch
- torchvision
- tabulate
- datetime
- argprse
- matplotlib
- os

## Usage

1. Experiment

```
python Project_model.py --batch 128 --checkpoint_dir path/to/checkpoint_dir
```
`batch` means the batch size of network, there are several other parameters you change. 
For other options, please refer helps: `python Project_model.py -h`.
When you run the code for the first time, the dataset will be downloaded automatically.

2. Plot result

```
python Plot.py --parameter 'lr' --path path/to/experiment/logfile
```
`parameter` means the specific parameter experiment you want to comparing.
`path` lead to the file contain experiment log file.

3. others
   Kaggle_prediction.ipynb is only for predict the unlabeled images on kaggle, and output.csv contains the prediction result.

## Note
The device in this project is: device = 'cuda' if torch.cuda.is_available() else 'mps', which could only work on GPUs and Apple M1/M1 Pro. If you have a specific device to use, please change the device.
