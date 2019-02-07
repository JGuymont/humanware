# How to use

1. Execute `python split_data.py` to split the training set metadata in 
   training/validation/test (default 0.7/0.2/0.1)
1. Put your model's class in the folder models.
1. Put all the settings you want in a ini file in **config/**
   (check ***config/example.ini** for an example of configuration).
1. Execute `python main.py config/<name>.ini` to start the training of your model.
