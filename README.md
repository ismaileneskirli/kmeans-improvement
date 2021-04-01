# K-MEANS ALGORITHM IMPLEMENTATION & IMPROVEMENT
Here are my works for improving kmeans algorithm.

## How K-means work ?

1. Select k points as initial centroids from dataset. (You need to know how many clusters to make)
2. Find the similarity measure between the data point and the identified k-cluster centroids.
3. Assign each data point to the cluster based on the distance calculated.


## First possible upgrade  : Choosing the right k ?

When we know how many cluster are required, thats perfect. But generally it is not the case.
Therefore figuring out the right K solves a great problem. Elbow method is one way to figure out optimal k .

###  Elbow Method

The method plots the various of cost with varying the value of K.
The point at which the distortion declines is the optimal k value.

Implementation of elbow method :
https://pythonprogramminglanguage.com/kmeans-elbow-method/


## Second Upgrade : Not dependant of initial centroid

Kmean algorithm is very dependant on initial centroids selected. To eliminate this randomness, we will define a tolerance value and max iteration variable.
Until these values are met we will take means of clusters and substract from initial centroids.





## Useful resources :

https://learnai1.home.blog/2020/06/19/k-means/
Github repo for data file : https://github.com/Hello-World-Blog/Artificial-Intelligence/blob/master/K-Means/iris.csv
For choosing initial centroids
https://en.wikipedia.org/wiki/K-means%2B%2B