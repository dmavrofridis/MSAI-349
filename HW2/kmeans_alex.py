import math
from euclidean_distance import euclidean
from cosine_similarity import cosim
from random import randrange
from helper import sum_lists


def accuracy_checker(second):
    temp = {i: [] for i in range(10)}
    for cluster in second:
        if not second[cluster]:
            second[cluster].append(0)

        count = 0
        if second[cluster]:
            most_common = max(second[cluster], key=second[cluster].count)

        for element in second[cluster]:
            if element == most_common:
                count += 1
        if len(second[cluster]) > 1:
            temp[cluster] = count / len(second[cluster])
        else:
            temp[cluster] = 0
    return temp


def overall_accuracy(candidate):
    number = 0
    for num in candidate:
        number += candidate[num]
    return number / 10


def kmeans_new(train, query, metric):

    best = {i: [] for i in range(10)}
    initial = 0
    while overall_accuracy(accuracy_checker(best)) < 65 and initial < 40:
        try:
            initial += 1

            X = []
            for i in range(len(train)):
                X.append(train[i][1])
            clusters = {i: [] for i in range(10)}

            cluster_indexes = []
            length_of_data = len(train)
            i = 0
            multiplier = 1
            old_cluster_indexes = cluster_indexes
            cluster_row_number = {i: [] for i in range(10)}

            while i <= length_of_data:
                cluster_point = randrange(i, (length_of_data / (10) * multiplier))
                cluster_indexes.append(cluster_point)
                i += length_of_data / (10)
                multiplier += 1
            first_iteration = True

            for i in range(length_of_data):
                distances = {}
                for j in range(10):
                    dist = euclidean(train[cluster_indexes[j]][1], train[i][1])
                    distances[j] = dist
                min_dist = min(distances, key=distances.get)
                """     PROBLEM HERE    """
                clusters[min_dist].append(train[i][0])
                """     PROBLEM HERE    """
                cluster_row_number[min_dist].append(i)

            # print(clusters)
            current_clust_acc = accuracy_checker(clusters)
            overall = overall_accuracy(current_clust_acc)
            if overall > overall_accuracy(accuracy_checker(best)):
                best = clusters
            print(best)
            if overall_accuracy(accuracy_checker(best)) > 64:
                return best

            # clusters_average = {}
            clusters_average = {i: [] for i in range(10)}
            for i in range(10):
                for feature in range(len(train[0][1])):
                    su = 0
                    count = 0
                    for sample in range(len(clusters[i])):
                        #   print(train[clusters[i][sample]][1][feature])
                        su += train[clusters[i][sample]][1][feature]
                        count += 1
                    clusters_average[i].append(int(su / count))

            old_clusters_average = clusters_average
            first_iteration = True
            times_to_average = 0
            # while sum([euclidean(clusters_average[i],  old_clusters_average[i]) for i in range(10)]) or first_iteration == True or i <100 :
            # while  sum([euclidean(clusters_average[i],  old_clusters_average[i]) for i in range(10)]) or first_iteration  or i <15:
            while (
                overall_accuracy(current_clust_acc(best)) < 60 and times_to_average < 10
            ):
                times_to_average += 1

                try:

                    first_iteration_ = False
                    old_clusters_average = clusters_average

                    clusters = {i: [] for i in range(10)}

                    for i in range(length_of_data):
                        distances = {}
                        for j in range(10):
                            dist = euclidean(clusters_average[j], train[i][1])
                            distances[j] = dist
                        min_dist = min(distances, key=distances.get)
                        """     PROBLEM HERE    """
                        clusters[min_dist].append(train[i][0])
                        """     PROBLEM HERE    """
                        cluster_row_number[min_dist].append(i)

                    current_clust_acc = accuracy_checker(clusters)
                    print(clusters)
                    overall = overall_accuracy(current_clust_acc)
                    if overall > overall_accuracy(accuracy_checker(best)):
                        best = clusters
                    print(best)
                    if overall_accuracy(accuracy_checker(best)) > 64:
                        return best

                    clusters_average = {i: [] for i in range(10)}
                    for i in range(10):
                        for feature in range(len(train[0][1])):
                            su = 0
                            count = 0
                            for sample in range(len(clusters[i])):
                                #   print(train[clusters[i][sample]][1][feature])
                                su += train[clusters[i][sample]][1][feature]
                                count += 1
                            if count != 0:
                                clusters_average[i].append(int(su / count))
                except:
                    continue
        except:
            continue

    return best


def kmeans_new3(train, query, metric):
    X = []
    for i in range(len(train)):
        X.append(train[i][1])
    clusters = [[] for i in range(10)]
    cluster_indexes = []
    length_of_data = len(train)
    i = 0
    multiplier = 1
    old_cluster_indexes = cluster_indexes

    while i <= length_of_data:
        cluster_point = randrange(i, (length_of_data / (10) * multiplier))
        cluster_indexes.append(cluster_point)
        i += length_of_data / (10)
        multiplier += 1
    first_iteration = True

    for i in range(length_of_data):
        distances = {}
        for j in range(10):
            dist = euclidean(train[cluster_indexes[j]][1], train[i][1])
            distances[j] = dist
        min_dist = min(distances, key=distances.get)
        clusters[min_dist].append(i)
    clusters_average = {i: [] for i in range(10)}

    clusters_average = {}

    for i in range(10):
        for feature in range(len(train[0][1])):
            su = 0
            count = 0
            for sample in range(len(clusters[i])):
                print(train[clusters[i][sample]][1][feature])
                su += train[clusters[i][sample]][1][feature]
                count += 1
            clusters_average[i].append(su / count)

    while (
        sum(
            [euclidean(clusters_average[i], old_clusters_average[i]) for i in range(10)]
        )
        or first_iteration == True
    ):
        first_iteration_ = False
        old_cluster_indexes = cluster_indexes

        clusters = {i: [] for i in range(10)}

        for i in range(length_of_data):
            distances = {}
            for j in range(10):
                dist = euclidean(train[cluster_indexes[j]][1], train[i][1])
                distances[j] = dist
            min_dist = min(distances, key=distances.get)
            clusters[min_dist].append(i)
        clusters_average = {i: [] for i in range(10)}
        print(clusters_average)

        for i in range(10):
            for feature in range(len(train[0][1])):
                su = 0
                count = 0
                for sample in range(len(clusters[i])):
                    print(train[clusters[i][sample]][1][feature])
                    su += train[clusters[i][sample]][1][feature]
                    count += 1
                clusters_average[i].append(su / count)

        return clusters_average


#        for i in clusters:
#           clusters_mean[i]= train[clusters[i][j]] for j in  clusters[i]


# given a list of centroids, label a point
# centroids is a list of centroids
# labels is the list of each centroid such that the label of centroids[i] is labels[i]
# point is the point to be labelled
# dist_type is the type of distance to be used in calcualtion
def label_point(centroids, c_labels, point, dist_type):
    # setup
    length = len(c_labels)
    min_dist = float("inf")
    min_label = None

    # check if distance between centroid and point is below minimum recorded distance
    # if so, change min_label and min_dist to show this
    for i in range(length):
        dist = float("inf")
        # print(centroids[i], point)
        if dist_type == "euclidean":
            dist = euclidean(centroids[i], point)
        elif dist_type == "cosine":
            dist = cosim(centroids[i], point)
        # print("dist from ", i, ":", dist, "cen:" , centroids[i],"point:" ,point)
        if dist < min_dist:
            min_label = c_labels[i]
            min_dist = dist
    # print(min_label)
    return min_label


# given a list of points determine new location for centroid
def centroid(points):
    if len(points) == 0:
        return None
    new_centroid = []
    # print(points)
    length = len(points)
    dim = len(
        points[0]
    )  # calculates the number of dimensions of every point in the points list
    for i in range(dim):
        new_centroid.append(0)
        # adds each value to centroid
        for p in points:
            new_centroid[i] += p[i]
        new_centroid[i] /= length  # finds the average
    return new_centroid


# compares old centroids to new centroids,
# if they are the same or max iterations has been reached return true, else false
def stop(old, new, i, n):
    return old == new or n > i


# Initialing random centroid based on number of features and k.
def random_centroids(length_of_vector, k):
    random_centroids = []
    for i in range(k):
        cent = []
        for j in range(length_of_vector):
            rand = randrange(10)
            if rand <= 7:
                rand = 0
            else:
                rand = 1
            cent.append(rand)
        random_centroids.append(cent)
    return random_centroids


def remove_labels(dataset):
    labelless = []
    for v in dataset:
        labelless.append(v[1])
    return labelless


def add_label(vector, label):
    return [label, vector]


def kmeans(train, query, metric):
    k = 5
    max_iterations = 100
    iterations = 0

    num_points = len(train)
    dim = len(train[0][1])

    labels = []
    for i in range(k):
        labels.append(i)

    old_centroids = None
    new_centroids = random_centroids(dim, k)
    points = remove_labels(train)
    point_count = []

    while not stop(old_centroids, new_centroids, max_iterations, iterations):

        old_centroids = new_centroids
        new_centroids = []
        point_count = []
        # initializes variables for this step
        for i in range(k):
            cent = []
            for j in range(dim):
                cent.append(0)
            new_centroids.append(cent)
            point_count.append(0)

        # determines the label of a point and adds point to new centroids
        for p in points:
            label = label_point(old_centroids, labels, p, metric)
            point_count[label] += 1

            new_centroids[label] = sum_lists(new_centroids[label], p)

        # divides new centroids by the number of points associated with the centroid
        for i in range(k):

            if point_count[i] != 0:
                for j in range(dim):
                    new_centroids[i][j] /= point_count[i]
            else:
                new_centroids[i] = old_centroids[i]

        iterations += 1
        print(iterations)

        output = []
        for p in points:
            label = label_point(new_centroids, labels, p, metric)
            output.append(label)
            # output.append(add_label(p, label))
    return output


# returns a list of labels for the query dataset based upon observations in the train dataset.
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".
# All hyper-parameters should be hard-coded in the algorithm.
def old_kmeans(train, query, metric):
    # hyperparameters
    k = 10
    max_iterations = 100
    iterations = 0
    old_centroids = None

    labels = []
    for i in range(k):
        labels.append(i)

    # length of each vector in the dataset
    length_of_vector = len(train[0][1])
    # length of the total dataset (eg. how many numbers)
    length_of_dataset = len(train)
    # Initialing random centroid
    new_centroids = random_centroids(length_of_vector, k)
    points = remove_labels(train)
    point_labels = []
    while not stop(old_centroids, new_centroids, max_iterations, iterations):
        # reset the old_centroid
        # print(old_centroids," - > " ,new_centroids)
        if iterations != 0:
            for i in range(len(old_centroids)):
                pass
                # print(euclidean(old_centroids[i], new_centroids[i]))
        old_centroids = new_centroids

        # print(old_centroids)
        # vector which holds the different labels we collect
        point_labels = []
        # print(length_of_vector)

        # determine labels for each "number" in the dataset with min distance
        for p in points:
            point_labels.append(label_point(new_centroids, labels, p, metric))
        # print(old_centroids, point_labels, points)
        # print("")
        # Get the new centroids based on new label
        new_centroids = []
        for i in range(k):
            point_assoc = []
            for j in range(len(points)):
                if point_labels[j] == i:
                    point_assoc.append(points[j])
            # print(point_assoc)
            # print(i, "points", len(point_assoc))
            c = centroid(point_assoc)
            # print(c, point_assoc)
            if c is not None:
                new_centroids.append(c)
            else:
                new_centroids.append(old_centroids[i])

        # print(new_centroids)
        iterations += 1
        print(iterations)
    # unite the data
    output = []
    for i in range(len(points)):
        # print(point_labels[i])
        output.append(add_label(points[i], point_labels[i]))

    return output
