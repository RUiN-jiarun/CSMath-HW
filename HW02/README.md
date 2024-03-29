# HW02 Visualization of Hand-written Digits

## Environment
* Python 3.9
* Jupyter Notebook
* numpy, matplotlib
* pytorch 1.12+

## Basic Goal
* First extract target data from the original datasets:  
    ![](img/sample_3.png)  
* Do PCA and visualize the eigen feature of all '3's:  
    ![](img/mean_3.png)  
* Visualize 2 principal components and the corresponding '3's:  
<figure class="half">
    <img src="img/pca_scatter.png" width="400">
    <img src="img/pca_eigen.png" width="400">
</figure>  

## Advanced Goal
Apply t-SNE to all '3's and extract 2 principal components.  
The data has been dimensionally reduced by a simple PCA into 10 components.  
The training loss curve:  
<img src="img/tsne_loss.png" width="300">
<figure class="half">
    <img src="img/tsne_scatter.png" width="400">
    <img src="img/tsne_eigen.png" width="400">
</figure>  

## Bonus
Process the whole dataset in deep learning methods and visualize '3's.  
### MLP
I use a three-layer MLP to handle this partition task. The network architecture is very simple:  
```python
self.layer1 = nn.Linear(input, 512)
self.relu1 = nn.ReLU()
self.layer2 = nn.Linear(512, 128)
self.relu2 = nn.ReLU()
self.layer3 = nn.Linear(128, classes)
self.softmax = nn.Softmax()
```  
The training loss curve and accuracy on testing dataset:  
<figure class="half">
    <img src="img/mlp_loss.png" width="400">
    <img src="img/mlp_acc.png" width="400">
</figure>  

The visualization results:  
<figure class="half">
    <img src="img/mlp_scatter.png" width="400">
    <img src="img/mlp_eigen.png" width="400">
</figure>  