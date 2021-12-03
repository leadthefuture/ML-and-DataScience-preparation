class KNeighborsClassifier:
    '''Implementation from scratch of the KNN algorithm'''

    def __init__(self, n_neighbors=5, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = self.set_metric(metric)

    def set_metric(self,metric):
        '''Set the metric used for calculating the distances
        between points
        
        Args:
            metric (str) : distance metric provided by the user
        Returns:
            return the given metric if it has been implemented, otherwise
            raise an Error
        '''    
        if metric in ['euclidean']:
            return metric
        else:
            raise ValueError('{} has not been implemented yet - refer to euclidean at the moment'.format(metric))


if __name__ == '__main__':
    knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')