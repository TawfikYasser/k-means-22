import numpy as np
import pandas as pd

data = pd.read_csv('Power_Consumption.csv')
# we need the columns from 1 -> 7 (Features)
np.set_printoptions(suppress=True)
Values = data.values
Values = Values[:, 1:]
print(Values)

K = int(input('Please enter the number of cluster '))

distance_measure = input('Please type 1 for Euclidean and 2 for Manhattan ')

def k_means_clustering(Values, K, distance_measure):
    # Step 1: Initialize random cluster centers (centroids) based on the K
    centroids = data.sample(n=K).values
    print(centroids)
    centroids = centroids[:, 1:]
    print(centroids)
    # Step 2: Iterate over the data rows, calculate the distance between each row and each one of the random centroids
    # and assign the row to the closest centroid, until convergence which means that the centroids don't change
    isDifference = 1
    clusters = {}
    while isDifference:
        for row_index, row_itself in enumerate(Values):
            # Iterating over the rows
            list_of_distances = []
            for cluster_index, centroid in enumerate(centroids):
                # Iterating over the centroids
                if distance_measure == 1:
                    # Euclidean distance
                    distance = np.sqrt( (centroid[0]-row_itself[0])**2 + (centroid[1]-row_itself[1])**2 + (centroid[2]-row_itself[2])**2 +
                                (centroid[3]-row_itself[3])**2 + (centroid[4]-row_itself[4])**2 +
                                (centroid[5]-row_itself[5])**2 + (centroid[6]-row_itself[6])**2 )
                else:
                    # Manhattan distance
                    distance = ( abs(centroid[0]-row_itself[0]) + abs(centroid[1]-row_itself[1]) + abs(centroid[2]-row_itself[2]) 
                    + abs(centroid[3]-row_itself[3]) + abs(centroid[4]-row_itself[4]) + abs(centroid[5]-row_itself[5]) + abs(centroid[6]-row_itself[6]) )
                list_of_distances.append(distance)

            # Assigning the row_index to the closest centroid
            clusters[row_index] = list_of_distances.index(min(list_of_distances))
            # print(clusters)
        
        # Step 3: Calculate the new centroids based on the new cluster assignments
        new_centroids = pd.DataFrame(Values).groupby(by=clusters).mean().values
        print(f'New Centroids {new_centroids}')
        # Step 4: Check if the centroids have changed, if not then we are done
        print(f'Centroids {centroids}')
        if np.count_nonzero((centroids)-(new_centroids)) == 0:
            isDifference = 0
        else:
            centroids = new_centroids
    return clusters, centroids


clusters, centroids = k_means_clustering(Values, K, distance_measure)
print('#'*50)
print(f'Final Clusters: {clusters}')
print(f'Final Centroids: {centroids}')

row_to_cluster_distance = {}

for index, row in enumerate(Values):
    print(f'Row: {row} belongs to cluster {clusters[index]}')
    print(f'Cluster: {clusters[index]}')
    desired_cluster = clusters[index]
    print(f'Centroid: {centroids[desired_cluster]}')
    dist = np.sqrt( (centroids[desired_cluster][0]-row[0])**2 + (centroids[desired_cluster][1]-row[1])**2 + (centroids[desired_cluster][2]-row[2])**2 +
            (centroids[desired_cluster][3]-row[3])**2 + (centroids[desired_cluster][4]-row[4])**2 +
            (centroids[desired_cluster][5]-row[5])**2 + (centroids[desired_cluster][6]-row[6])**2 )
    row_to_cluster_distance[index, dist] = desired_cluster


sort_by_value = dict(sorted(row_to_cluster_distance.items(), key=lambda item: item[1]))
import collections

new_dict = collections.defaultdict(list)

for i, j in sort_by_value.items():
    new_dict[j].append(i)
    print(i, j)

for k, val in new_dict.items():
    print(k, len(val))
    sorted_list = sorted(val, key = lambda x: x[1])
    print(f'Sorted list for cluster {k} is {sorted_list}')
    print(f'Cluster {k} has {int(len(val) * 0.95)} rows inside the range')
    print(f'Outliers of cluster {k} are: ')
    for i in sorted_list[int(len(val) * 0.95):]:
        print(i)
