import math
#from silhouette import NumberOfClusters; #For pow and sqrt
import sys;
import sklearn
from random import shuffle, uniform;
#from sklearn.metrics import silhouette_score
from sklearn import cluster


#### Preparing the Data
def ReadData(filename):
    f = open(filename, 'r')
    #each line is an element of lines array.
    lines = f.read().splitlines()
    f.close()
    #print(lines)
    items = []
    for i in range (1, len(lines)):
        #line is an array where each feature is different feature.
        line = lines[i].split(',')
        itemFeatures = []

        for j in range(len(line) -1):
            ## convert feature ( string ) to float
            feature = float(line[j])

            ## add feature to array, not the last feauture.
            itemFeatures.append(feature)
        items.append(itemFeatures)

    shuffle(items)

    return items
## returns the mins and max of the each attribute.
def FindColMinMax(items):
    n = len(items[0]);
    minima = [sys.maxsize for i in range(n)];
    maxima = [-sys.maxsize -1 for i in range(n)];

    for item in items:
        for f in range(len(item)):
            if (item[f] < minima[f]):
                minima[f] = item[f]

            if (item[f] > maxima[f]):
                maxima[f] = item[f];

    return minima,maxima;

## returns an array of len k, each element of array is an array of centroids len(attribute number of items)
# array = [[4],[4],[4 eleman]]
def InitializeMeans(items, k, cMin, cMax):

    # Initialize means to random numbers between
    # the min and max of each column/feature
    f = len(items[0]); # number of features
    means = [[0 for i in range(f)] for j in range(k)];

    for mean in means:
        for i in range(len(mean)):

            #TODO aşağıdaki kısmı fonksiyona alabilirsin.
            # Set value to a random float
            # (adding +-1 to avoid a wide placement of a mean)
            ## random point within given interval
            mean[i] = uniform(cMin[i]+1, cMax[i]-1);
            #cprint(mean[i])

    return means;# centroids.


# items = ReadData("iris.txt")
# min,max = FindColMinMax(items)

# print(min,max)

# means = InitializeMeans(items, 3, min, max)

# print(means)


# lines = ReadData("iris.txt")
# print("asd")
# print(lines)

# TODO : Possible upgrade here : maybe choosing another distance method ?
def EuclideanDistance(x, y):
    S = 0; # The sum of the squared differences of the elements
    for i in range(len(x)):
        S += math.pow(x[i]-y[i], 2)

    #The square root of the sum
    return math.sqrt(S)


def UpdateMean(n,mean,item):
    for i in range(len(mean)):
        m = mean[i];
        m = (m*(n-1)+item[i])/float(n);
        mean[i] = round(m,3);

    return mean;


def FindClusters(means,items):
    clusters = [[] for i in range(len(means))]; #Init clusters

    for item in items:
        #Classify item into a cluster
        index = Classify(means,item);

        #Add item to cluster
        clusters[index].append(item);

    return clusters;


###_Core Functions_###
## Classify returns the class of item which is index -> 0,1,2
def Classify(means,item):
    #Classify item to the mean with minimum distance

    minimum = sys.maxsize;
    index = -1;

    for i in range(len(means)):
        #Find distance from item to mean
        dis = EuclideanDistance(item,means[i]);

        if(dis < minimum):
            minimum = dis;
            index = i;

    return index;

def CalculateMeans(k,items,maxIterations=100000):
    #Find the minima and maxima for columns
    cMin, cMax = FindColMinMax(items);

    #Initialize means at random points
    means = InitializeMeans(items,k,cMin,cMax);
    print(means)

    #Initialize clusters, the array to hold
    #the number of items in a class
    clusterSizes = [0 for i in range(len(means))];

    #An array to hold the cluster an item is in
    belongsTo = [0 for i in range(len(items))];

    #Calculate means
    for e in range(maxIterations):
        #If no change of cluster occurs, halt
        noChange = True;
        for i in range(len(items)):
            item = items[i];
            #Classify item into a cluster and update the
            #corresponding means.

            ## return which class ite item belongs.
            index = Classify(means,item);

            clusterSizes[index] += 1;
            means[index] = UpdateMean(clusterSizes[index],means[index],item);

            #Item changed cluster
            if(index != belongsTo[i]):
                noChange = False;

            belongsTo[i] = index;

        #Nothing changed, return
        if(noChange):
            break;

    return means;




# Function to return initial centroids.
# For the moment it is already not random, i take the min and max of each attribute and subtract -1 to avoid cases.
# TODO : replace this function with Initialize means !
def choose_initial_centroids():
    ## Choose first centroid minima + 1,
    # choose subsequent centroids from the remaining data point based on the distance
    # with weighted probability distribution
    #steps above to be repeated k -1 times.

    return initial_centroids

# return silhouette scores for given clustered items
def silhouette_score(clusters):
    ## TODO : implement this.



    return silhouette_score



######################## At the end of the project i will only call this function to cluster given dataset.############
#TODO
# Kmeans function without giving k as a parameter, it takes only data to be classed as parameter.
def Kmeans(items):
    ### pseudo code, working on it for the moment. :
    NumberOfClusters = range(2,20)
    silhouette_scores = []
    for i in range(0,NumberOfClusters):
        means = CalculateMeans(i,items)
        clusters = FindClusters(means,items);
        silhouette_scores.append(silhouette_score(clusters))


    ## Brute force silhouette scores for a given range then return clusters with best silhoutte score.
    best_means = CalculateMeans(max(silhouette_scores),items)
    best_clusters = FindClusters(best_means, items)


    return best_clusters

###_Main_###
def main():
    items = ReadData('iris.txt');
    #print(items)


    # TODO : Choose k with function defined.
    k = 3;

    means = CalculateMeans(k,items);
    #print(means) # average of attributes for each cluster(class)
    clusters = FindClusters(means,items);
    print(clusters)
    #print (clusters[0]);

    # able to classify items for the moment. will compare with choosing better initial points.
    newitem1 = [5.1,3.5,1.4,0.2]
    newitem2 = [7.0,3.2,4.7,1.4]
    newitem3 = [5.7,2.5,5.0,2.0]
    # print(Classify(means,newitem1))
    # print(Classify(means,newitem2))
    # print(Classify(means,newitem3))
    #newItem = [5.4,3.7,1.5,0.2];
    #print Classify(means,newItem);

if __name__ == "__main__":
    main();