# Humanware-block1

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

## SCP
```
scp user23@helios.calculquebec.ca:/rap/jvb-000-aa/COURS2019/etudia
nts/data/humanware/SVHN/train.tar.gz C:\Users\ACER\university\IFT6759\Humanware-block1\data\SVHN

scp user23@helios.calculquebec.ca:/rap/jvb-000-aa/COURS2019/etudiants/data/humanware/SVHN/train_metadata.pkl C:\Users\ACER\university\IFT6759\Humanware-block1\data\SVHN
```