conda create -n FedLaAvg python==3.6.9
conda activate FedLaAvg

mkdir FedLaAvg
cd FedLaAvg || exit
git clone https://github.com/mikudehuane/FedLaAvg scripts
pip install -r scripts/requirements.txt

echo "project_dir = \"$(pwd)\"" > scripts/config.py

mkdir cache data
python scripts/train/model.py

mkdir -p models/glove/
wget http://nlp.stanford.edu/data/glove.twitter.27B.zip -o models/glove/glove.twitter.27B.zip
unzip models/glove/glove.twitter.27B.zip

mkdir raw_data/Sentiment140
wget http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip -o raw_data/Sentiment140/sentiment140.zip
unzip raw_data/Sentiment140/sentiment140.zip

