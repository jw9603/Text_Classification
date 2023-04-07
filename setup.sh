conda create -n ntc python=3.8 ipykernel
source activate ntc
conda install pytorch=1.9.0 torchvision=0.10.0 torchaudio=0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install torch_optimizer
conda install -c pytorch torchtext==0.10.0
conda install ignite -c pytorch