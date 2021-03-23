import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


class Kmeans :
    #constructor to initialize the model with a default value of “k” cluster to 8
    # maximum iterations to take when centroids not coincide to 100.
    def __init__(self, k = 8, max_iter = 100):
        self.k = k
        self.max_iter = max_iter
        print("Initialized k with : " ,k)

    # Calculating Distance :: Any distance metric can be used
    def euclidDistance (self, x1, x2):
    # Takes the centroid vector and datapoint vector, performs substraction between vectors.
    # Squares the result
    # sum all square and the apply sqrt.
        distance = np.sqrt(np.sum(np.square(x1-x2), axis=1))


    # Takes data and forms clusters based on similarity.
    def fit(self, data):
        self.centroids = []

        #first k points in the dataset are our centroids.
        for i in range (self.k):
            ## data.iloc Purely integer-location based indexing for selection by position.
            self.centroids.append(data.iloc[i].to_numpy())


        for iteration in range(self.max_iter):
            # Defining a dict for our classes
            self.classes = {}

            for cluster in range (self.k):
                self.classes[cluster] = []

            #Based on the distance of a data point and centroid, assign each point to a neares cluster.
            for data_point in range (len(data)):
                #Finds the similarity measure between the data point and the identified k-cluster centroids, distances is an array lenght k.
                distances = self.euclidDistance(self.centroids,data.iloc[data_point].to_numpy())
                #Returns the indices of the minimum values along an axis, so it returns the indice of closest centroid to the data point.
                classification = np.argmin(distances)
                self.classes[classification].append(data.iloc[data_point])

            previousCentroids = np.array(self.centroids)
            #Take the average of the cluster datapoints to recalculate the centroids, new centroids are the average of each classs.
            for classification in self.classes:
                self.centroids[classification] = np.average(self.classes[classification], axis=0)


            optimal = True
            currentCentroids = np.array(self.centroids)

            ## Convergence is the difference between centroids, the sum of ( currentCentroid - previousCentroid) difference of centroids should be almost 0.
            tolerance = 0.0001
            if np.sum((currentCentroids - previousCentroids)/ previousCentroids * 100) > tolerance:
                optimal = False # if difference is more than this epsilon value then it is not an optimal centroid and it will keep changing over iterations.


            # if we found the optimal centroids then there is no need to keep iterating. It will not change more than the tolarence value defined.
            if optimal:
                break


def main():
    ## Data preperation.
    file = './iris.csv'
    data = pd.read_csv(file)
    data.head()


    clf = Kmeans(3)



    X = data.iloc[:,:-1]

    ## Parameters x vector, y vector, hue = Grouping variable that will produce points with different colors.
    # Can be either categorical or numeric, although color mapping will behave differently in latter case.
    sb.scatterplot(X.iloc[:,0] , X.iloc[:,1] , hue = data.iloc[:,-1])