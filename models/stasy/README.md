# STaSy: Score-based Tabular Data Synthesis
This code is the official implementation of "STaSy: Score-based Tabular Data Synthesis".

## Requirements
Run the following to install requirements:
```setup
conda env create --file requirements.yaml
```
## Usage
* Train, fine-tune, and evaluate our STaSy through `main.py`:
```sh
main.py:
  --config: Training configuration.
  --mode: <train|fine_tune|eval>: Running mode: train or fine_tune or eval
  --workdir: Working directory
```

## Training
* You can train our STaSy with SPL from scratch by run:
```bash
python main.py --config configs/shoppers.py --mode train --workdir stasy
```
* To fine-tune the model trained with SPL by run:
```bash
python main.py --config configs/shoppers.py --mode fine_tune --workdir stasy --config.optim.lr 2e-07
```

## Evaluation
* You can download pretrained model from the anonymous link [here](https://drive.google.com/drive/folders/12voQqxsFwGSznVR6_iP8t0EWl5nOI35n?usp=sharing).
* Downloaded checkpoint should be in 'stasy/checkpoints/'.
* By run the following script, you can reproduce our experimental result: 
    binary classification result of STaSy on Shoppers in Table 11. 
```bash
python main.py --config configs/shoppers.py --mode eval --workdir stasy  
```
