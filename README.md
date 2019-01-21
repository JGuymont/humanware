# Humanware-block1

## Running the experiment
```
python split_data.py --train_pct 0.7 --valid_pct 0.2 --test_pct 0.1
python main.py --model [ModelPaper, ConvNet]
```
Note: the images need to be in `./data/SVNH/train/`

## Data
```
cd Humanware-block1
cp '/rap/jvb-000-aa/COURS2019/etudiants/data/humanware/SVHN/train_metadata.pkl' './data/SVHN/'
cp '/rap/jvb-000-aa/COURS2019/etudiants/data/humanware/SVHN/train.tar.gz' './data/SVHN/'
tar -xzf $HOME'/digit-detection/data/SVHN/train.tar.gz' -C './data/SVHN/'
```

## Container
```
source /rap/jvb-000-aa/COURS2019/etudiants/common.env
echo 'source /rap/jvb-000-aa/COURS2019/etudiants/common.env' >> ~/.bashrc
singularity shell --nv /rap/jvb-000-aa/COURS2019/etudiants/ift6759.simg
```
