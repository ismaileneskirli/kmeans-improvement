# K-MEANS ALGORITHM IMPLEMENTATION & IMPROVEMENT
Here are my works for improving kmeans algorithm.

## How K-means work ?

1. Select k points as initial centroids from dataset. (You need to know how many clusters to make)
2. Find the similarity measure between the data point and the identified k-cluster centroids.
3. Assign each data point to the cluster based on the distance calculated.


## First possible upgrade  : Choosing the right k ?

When we know how many cluster are required, thats perfect. But generally it is not the case.
Therefore figuring out the right K solves a great problem.

###  Elbow Method

The method plots the various of cost with varying the value of K.
The point at which the distortion declines is the optimal k value.





## Useful resources :

https://learnai1.home.blog/2020/06/19/k-means/
Github repo for data file : https://github.com/Hello-World-Blog/Artificial-Intelligence/blob/master/K-Means/iris.csv