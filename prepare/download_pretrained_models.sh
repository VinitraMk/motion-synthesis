mkdir -p checkpoints/
cd checkpoints/
mkdir -p model/
cd model/
echo -e "The pretrained models will stored in the 'checkpoints/model' folder\n"
# humanml3d-vae chkpoint
gdown "https://drive.google.com/uc?id=1vPM98-yzuDHMhH_JPyDK61jSLqWgeexj"
# vqvae_beta_0.1_full_v0.tar
gdown "https://drive.google.com/uc?id=1_va6ssU16OYcxFAwr7ZBmgdqsUQPBls8"
echo -e "Downloading done!"
