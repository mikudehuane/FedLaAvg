conda create -n FedLaAvg python==3.6.9
conda activate FedLaAvg
mkdir FedLaAvg
cd FedLaAvg || exit
git clone https://github.com/mikudehuane/FedLaAvg scripts
pip install -r scripts/requirements.txt
echo "project_dir = $(pwd)" > scripts/config.py

