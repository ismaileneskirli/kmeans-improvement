import math
#from silhouette import NumberOfClusters; #For pow and sqrt
import sys;
import sklearn
from random import shuffle, uniform;
#from sklearn.metrics import silhouette_score
from sklearn import cluster
from scipy import stats
import numpy as np
from difflib import SequenceMatcher

#### Preparing the Data
def ReadData(filename):
    f = open(filename, 'r')
    #each line is an element of lines array.
    lines = f.read().splitlines()
    f.close()
    #print(lines)
    items = []
    for i in range (0, len(lines)):
        #line is an array where each feature is different feature.
        line = lines[i].split(',')
        itemFeatures = []

        for j in range(len(line) -1):
            ## convert feature ( string ) to float
            feature = float(line[j])

            ## add feature to array, not the last feauture.
            itemFeatures.append(feature)
        itemFeatures.append(str(line[j+1]))
        items.append(itemFeatures)


    shuffle(items)

    return items

#### Preparing the Data, do not include label.
def ReadAttribute(filename):
    f = open(filename, 'r')
    #each line is an element of lines array.
    lines = f.read().splitlines()
    f.close()
    #print(lines)
    items = []
    for i in range (0, len(lines)):
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
        for f in range(len(item)-1):
            if (item[f] < minima[f]):
                minima[f] = item[f]

            if (item[f] > maxima[f]):
                maxima[f] = item[f];

    return minima,maxima;

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
    for i in range(len(x)-1):
        S += math.pow(x[i]-y[i], 2)

    #The square root of the sum
    return math.sqrt(S)


def UpdateMean(n,mean,item):
    for i in range(len(mean)):
        m = mean[i];
        m = (m*(n-1)+item[i])/float(n);
        mean[i] = round(m,3);

    return mean;

# For calculating the performance of my algorithm, each element is label.
def FindClusters(means,items):
    clusters = [[] for i in range(len(means))]; #Init clusters

    for item in items:
        #Classify item into a cluster
        index = Classify(means,item);

        #Add item to cluster
        clusters[index].append(str(item[4]));

    return clusters;
# For observing each item in cluster. Each elemant of cluster is attributes.
def return_clusters(means,items):
    clusters = [[] for i in range(len(means))]; #Init clusters

    for item in items:
        #Classify item into a cluster
        index = Classify(means,item);

        #Add item to cluster
        clusters[index].append((item[:4]));

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

    #Initialize means at random uniform points
    #means =InitializeMeans(items,k,cMin,cMax);
    #print(means)
    #Inıtialize means with proboability portionate to distances
    means = choose_initial_centroids(items,k,cMin,cMax);

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

def cal_average(num):
    sum_num = 0
    for t in num:
        sum_num = sum_num + t

    avg = sum_num / len(num)
    return avg

## returns an array of len k, each element of array is an array of centroids len(attribute number of items)
# array = [[4],[4],[4 eleman]]
def InitializeMeans(items, k, cMin, cMax):

    # Initialize means to random numbers between
    # the min and max of each column/feature
    f = len(items[0])-1; # number of features
    #print("f,k : ",f,k)
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

# Function to return initial centroids.
# For the moment it is already not random, i take the min and max of each attribute and subtract -1 to avoid cases.
# TODO : replace this function with Initialize means !
# use readAttributes function for items.
def choose_initial_centroids(items, k, cMin, cMax):
    ############### FIRST APPROACH : WEIGHTED PROBABILITY ########################
#     1- Choose one center uniformly at random from among the data points.
    means = []
    initial_mean = [0 for i in range(len(items[0])-1)]
    means.append(initial_mean)
    #print(items[0])
    for i in range(len(items[0])-1):
        initial_mean[i] = uniform(cMin[i]+1, cMax[i]-1);
#2-let D(x) denote the shortest distance from a data point to the closest center we have already chosen
# For each data point x, compute D(x), the distance between x and the nearest center that has already been chosen.
    for i in range (k):
        #print("initial_centroid secme sayısı", i)
        minimum_distances = []
        for item in items:
            distances = []
            for mean in means:
                distances.append(EuclideanDistance(item,mean))
            minimum_distances.append(min(distances))
        probability_array = []
        total_square = 0
        for distance in minimum_distances:
            total_square += distance*distance

        for distance in minimum_distances:
            probability_array.append(distance*distance / total_square)
    # 3-Choose one new data point at random as a new center, using a weighted probability
    # distribution where a point x is chosen with probability proportional to D(x)^2
    # (You can use scipy.stats.rv_discrete for that).
        xk = np.arange(150)
        custm = stats.rv_discrete(name='custm', values=(xk, probability_array))
        X = custm.rvs(size=1)
        #print(items[X[0]][:-1])
        means.append(items[X[0]][:-1])
    # Repeat Steps 2 and 3 until k centers have been chosen.
    # Now that the initial centers have been chosen, proceed using standard k-means clustering

    #print(means[:-k])
    return means

def max_abs_diff(arr):

    # To store the minimum and the maximum
    # elements from the array
    minEle = arr[0]
    maxEle = arr[0]
    for i in range(1, len(arr)):
        minEle = min(minEle, arr[i])
        maxEle = max(maxEle, arr[i])

    return (maxEle - minEle)

def find_max_difference_of_cluster_size(clusters):
    lengthArray = []
    for i in range(len(clusters)):
        lengthArray.append(len(clusters[i]))
        #print(i," boyutu : ",len(clusters[i]))
    return(max_abs_diff(lengthArray))




# return silhouette scores for given clustered items
def optimum_k(items,n):
    ## TODO : implement this.
    ## elemanın kendi clusterındaki her elemana olan uzaklığının ortalaması
    # bölü elemanın diğer clustardaki elemanlara olan uzaklığının ortalaması
    max_metrics = {}
    #print("size of items", len(items))
    for i in range (2,n+1):
        isOptimal = True
        #print("i : ",i)
        means = CalculateMeans(i,items)
        clusters = return_clusters(means,items)
        maxDifference = find_max_difference_of_cluster_size(clusters)
        #print("max difference of elements in each cluster is ",maxDifference)
        if(maxDifference > (len(items)/5)):
            #print("is not optimal Because. There should not be wide fluctuations between the size of the plots.")
            #print("max difference is ", find_max_difference_of_cluster_size(clusters))
            isOptimal= False

        for y in range(len(clusters)) :
            if(len(clusters[y]) < 1):
                #print(i, "is not an optimum k number, because kmeans left empty clusters.")
                isOptimal = False

        if(isOptimal):

            #print(clusters)
            evaluation_metrics = []
            #for each cluster
            for y in range(len(clusters)) :
                #print(len(clusters[y]))
                #for each item in cluster
                # when a_b is minimum we have the best k number.
                # for each item
                for j in range (len(clusters[y])):
                    if(len(clusters[y]) > 1):
                        total_distance_within_cluster = 0
                        average_distance_within_cluster = 0
                        total_distance_distinct_clusters = 0
                        average_distance_distinct_clusters = 0
                        evaluation_metric = 0
                        # for each point in cluster calculate
                        # the sum of distance between other points
                        #for each other item in cluster
                        for x in range (len(clusters[y])):
                            # if they are not the same item
                            if(x != j ):
                                #COHESION
                                total_distance_within_cluster += EuclideanDistance(clusters[y][j],clusters[y][x])
                        average_distance_within_cluster = total_distance_within_cluster/(len(clusters[y])-1)


                        #print("average_distance_within_cluster : ", average_distance_within_cluster)
                    # start calculating distance between other clusters SEPERATION
                    # for each other cluster
                        #counter = 0
                        neighboring_cluster = []
                        for b in range(len(clusters)):
                            # if they are not the same cluster

                            if (b != y):
                                # for each item in different clusters
                                if (len(clusters[b]) > 0):
                                    for b_x in range (len(clusters[b])):
                                        total_distance_distinct_clusters += EuclideanDistance(clusters[y][j],clusters[b][b_x])
                                        #counter += 1
                                    average_distance_distinct_clusters= total_distance_distinct_clusters/(len(clusters[b]))
                                    neighboring_cluster.append(average_distance_distinct_clusters)
                                else:
                                    # there is a cluster with no element in it.
                                    neighboring_cluster.append(sys.maxsize)
                        #average_distance_distinct_clusters = 0
                            total_distance_distinct_clusters = 0
                        average_distance_distinct_clusters = min(neighboring_cluster)
                    else:
                        average_distance_within_cluster = 1
                    evaluation_metric = (average_distance_distinct_clusters- average_distance_within_cluster) / max(average_distance_distinct_clusters, average_distance_within_cluster)
                    #evaluation_metric += average_distance_within_cluster / average_distance_distinct_clusters
                    evaluation_metrics.append(evaluation_metric)
                #print("max of cluster x",max(evaluation_metrics) )
            #average of each clusters silhouette score.
                #print(max(evaluation_metrics))
            max_metrics[i] = max(evaluation_metrics)
            # there should not be Presence of clusters with below-average silhouette scores
            #print("average of silhouette scores", cal_average(evaluation_metrics))
            # There should not be wide fluctuations between the size of the plots.
    maxSilhouette = 0
    index = 0
    for key in max_metrics.keys():
        #print("key is")
        if maxSilhouette  < max_metrics[key]:
            maxSilhouette = max_metrics[key]
            index = int(key)
    #print("optimum k is : ",index+1)
    #print(max_metrics)
    return index
# Function to compare two strings
def similarityRatio(a, b):
    return SequenceMatcher(None, a, b).ratio()

# Function to calculate how well clustering is
def calculate_accuracy(clusters):
    accuracies = []
    for cluster in clusters:
        #print(cluster)
        #print("-------")
        nbrOfSetosa = 0
        nbrOfVersicolor = 0
        nbrOfVirginica = 0
        for item in cluster:
            if item == 'Iris-setosa':
                nbrOfSetosa +=1
            if item == 'Iris-virginica':
                nbrOfVirginica +=1
            if item == 'Iris-versicolor':
                nbrOfVersicolor +=1
        accuracies.append(max([nbrOfSetosa,nbrOfVersicolor,nbrOfVirginica])/(nbrOfSetosa +nbrOfVersicolor +nbrOfVirginica) )
    #print (accuracies)
    return(accuracies)

######################## At the end of the project i will only call this function to cluster given dataset.############
#TODO
# Kmeans function without giving k as a parameter, it takes only data to be classed as parameter.
def Kmeans(items):
    #define k
    k=optimum_k(items,6)
    # define means, calculate means function uses chose_initial_centroid which i developed.
    means = CalculateMeans(k,items);
    #final clusters, you can print to observe them.
    clusters = FindClusters(means,items);
    # final results. accuracy is calculated by the proportion of same elements in cluster.
    return(calculate_accuracy(clusters))

# This function is written for report of my project.
def report_results(items):
    results = []
    counter = 0
    firstClusterTotal = 0
    secondClusterTotal = 0
    thirdClusterTotal = 0

    for i in range(20):
        results.append(Kmeans(items))

    for result in results:
        #print("secilen k sayisi : ", len(result))
        if (len(result) == 3):
            counter += 1
            firstClusterTotal += result[0]
            secondClusterTotal += result[1]
            thirdClusterTotal += result[2]
        print("Silhouette scores for each cluster", result)
    averageResults = [firstClusterTotal/counter, secondClusterTotal/counter,thirdClusterTotal/counter]
    print("Iris data setinde 3 class vardır. Algoritmam 20 defa calıstırılıp toplamda : ",counter, " kere 3 clustera ayırmıştır." )
    print("Dogru sayıda cluster bulma oranı yuzde ",counter*5)
    print("Denemeler sonucu doğru sayıda cluster olduğu zamanki ortalama her cluster için silhouette scorları : ", averageResults)

###_Main_###
def main():
    items = ReadData('iris.txt');
    report_results(items)
    #print(items[0])
    #Kmeans(items)
    # cMin,cMax = FindColMinMax(items)
    # k=3
    # choose_initial_centroids(items, k, cMin, cMax)
    # optimum_k(items,6)
    #print(items)

    ## Testing scenerios
    # able to classify items for the moment. will compare with choosing better initial points.
    # newitem1 = [5.1,3.5,1.4,0.2]
    # newitem2 = [7.0,3.2,4.7,1.4]
    # newitem3 = [5.7,2.5,5.0,2.0]
    # print(Classify(means,newitem1))
    # print(Classify(means,newitem2))
    # print(Classify(means,newitem3))
    #newItem = [5.4,3.7,1.5,0.2];
    #print Classify(means,newItem);


if __name__ == "__main__":
    main();