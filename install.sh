# Install all the necessary packages on Master

yum install -y tmux
yum install -y pssh
yum install -y python27 python27-devel
yum install -y freetype-devel libpng-devel
yum remove aws-cli
yum install aws-cli
wget https://bitbucket.org/pypa/setuptools/raw/bootstrap/ez_setup.py -O - | python27
easy_install-2.7 pip
easy_install py4j

pip2.7 install ipython==2.0.0
pip2.7 install pyzmq==14.6.0
pip2.7 install jinja2==2.7.3
pip2.7 install tornado==4.2

pip2.7 install numpy
# pip2.7 install matplotlib
pip2.7 install nltk
pip2.7 install boto
pip2.7 install xmltodict
pip2.7 install py4j
pip2.7 install gensim
pip install click

# EBS volume is mounted at /vol0, not enough room on root drive for NLTK data
mkdir /mnt/nltk_data
python27 -m nltk.downloader -d /mnt/nltk_data/ stopwords
echo 'export NLTK_DATA="/mnt/nltk_data"' >> ~/.bash_profile
echo 'export SPARK_HOME=/root/spark' >> ~/.bash_profile
echo 'export PATH=$PATH:$SCALA_HOME/bin:$SPARK_HOME/bin' >> ~/.bash_profile
echo 'export PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/build:$PYTHONPATH' >> ~/.bash_profile

# changing Python version
rm /usr/bin/python
ln -s /usr/bin/python2.7 /usr/bin/python

source ~/.bash_profile

# Install all the necessary packages on Workers

pssh -h /root/spark-ec2/slaves yum install -y python27 python27-devel
pssh -h /root/spark-ec2/slaves "wget https://bitbucket.org/pypa/setuptools/raw/bootstrap/ez_setup.py -O - | python27"
pssh -h /root/spark-ec2/slaves easy_install-2.7 pip

pssh -h /root/spark-ec2/slaves -v rm /usr/bin/python
pssh -h /root/spark-ec2/slaves -v ln -s /usr/bin/python2.7 /usr/bin/python

pssh -t 10000 -h /root/spark-ec2/slaves pip2.7 install numpy
pssh -h /root/spark-ec2/slaves pip2.7 install nltk
pssh -h /root/spark-ec2/slaves pip2.7 install gensim
pssh -h /root/spark-ec2/slaves pip2.7 install boto
pssh -h /root/spark-ec2/slaves pip2.7 install xmltodict
pssh -h /root/spark-ec2/slaves pip2.7 install py4j
