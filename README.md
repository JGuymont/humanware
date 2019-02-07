# Humanware-block1

## Models
Models are saved in the `models` directory. All the hyperparameters that are tuned should be access via a configuration file. Configuration file should be saved in the `config` folder. For example, let's say that there is a configuration file `./config/cnn.ini`, then the config file can be loader as 
```python
config = ConfigParser()
config.read('./config/cnn.ini')
```
and then pass to a model
```python
class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=config.getint("in_channels"),
            out_channels=config.getint("out_channels"),
            kernel_size=config.getint("kernel_size"))
        ...
```

## Running the experiment
1. The first time you run the experiment, you need to split the data into a training, validation, and testing set. The models are trained on the training set. The validation set is used to select the hyperparameters. The test set should only be used at the end when you are sure of your hyperparameters.
```
python split_data.py --train_pct 0.7 --valid_pct 0.2 --test_pct 0.1
```
2. To train a model, you first need to write a configuration file and save it in the `config` directory as `<config_name>.ini`. The the model you want to train is accessed via the config file with the parameter `name` in the `[model]` section. This name shoud be the name of the class of the model (you need to respect the capital letter).
```
python main.py config/<config_name>.ini
```
Note: 
1. the images need to be in `./data/SVNH/train/`
2. The model `LargeCNN` is the model describe in Goodfellow et al, 2013





