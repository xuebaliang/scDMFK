Single-cell transcriptome data clustering via multinomial modeling and adaptive fuzzy k-means algorithm
=====


Architecture
-----
![model](https://github.com/xuebaliang/scDMFK/blob/master/Architecture/Figure1.JPG)

Requirement
-----
Python 3.6

Tensorflow 1.14

Keras 2.2

Data availability
-----
The real data sets we used can be download in <a href="https://drive.google.com/drive/folders/1Mmbw2gPfgMzgy7ZDWV8Pqx2U2abiie8g">data</a>.

Quick start
-----
We use the dataset “Park” and scDMFK model to give an example. Please download all code files first. You just run the following code in your command lines:

python run.py --dataname "Park" --model "multinomial" --mode "indirect"

Then you will get the cluster result of “Park” dataset using scDMFK method in ten random seed. The median values of ARI and NMI are 0.832 and 0.776, respectively. Besides, you can also save the clustering label and low-dimensional latent representation for each cell to facilitate your other downstream analysis.
