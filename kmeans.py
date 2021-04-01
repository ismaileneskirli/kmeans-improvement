import math; #For pow and sqrt
import sys;
from random import shuffle, uniform;
from sklearn.metrics import silhouette_score

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
def InitializeMeans(items, k, cMin, cMax):

    # Initialize means to random numbers between
    # the min and max of each column/feature
    f = len(items[0]); # number of features
    means = [[0 for i in range(f)] for j in range(k)];

    for mean in means:
        for i in range(len(mean)):

            # Set value to a random float
            # (adding +-1 to avoid a wide placement of a mean)
            ## random point within given interval
            mean[i] = uniform(cMin[i]+1, cMax[i]-1);
            #cprint(mean[i])

    return means;


# items = ReadData("iris.txt")
# min,max = FindColMinMax(items)

# print(min,max)

# means = InitializeMeans(items, 3, min, max)

# print(means)


# lines = ReadData("iris.txt")
# print("asd")
# print(lines)

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


###_Main_###
def main():
    items = ReadData('iris.txt');

    k = 3;

    means = CalculateMeans(k,items);
    clusters = FindClusters(means,items);
    #print (means);
    #print (clusters[0]);


    newitem1= [5.1,3.5,1.4,0.2]
    newitem2= [7.0,3.2,4.7,1.4]
    newitem3 = [5.7,2.5,5.0,2.0]
    print(Classify(means,newitem1))
    print(Classify(means,newitem2))
    print(Classify(means,newitem3))
    #newItem = [5.4,3.7,1.5,0.2];
    #print Classify(means,newItem);

if __name__ == "__main__":
    main();