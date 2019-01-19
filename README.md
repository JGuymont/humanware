# Humanware-block1

## Data+fasd
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