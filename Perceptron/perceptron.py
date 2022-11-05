import numpy as np
import random
import copy

class Standard_Perceptron(object):
    '''
    Class for standard perceptron
    '''

    def __init__(self, num_epochs=10, *args, **kwargs) -> None:
        '''
        Initialize the standard perceptron with default values
        '''
        self.args         = args
        self.kwargs       = kwargs
        self.num_epochs   = num_epochs
        self.num_feats    = None
        self.num_train_ds = None
        self.w            = None
        self.lr           = None

    def train(self, train_ds, *args, **kwargs):
        '''
        Train the standard perceptron
        '''

        ## Get the number of features
        self.num_feats = len(train_ds[0]) - 1
        #print(f'num_feats: {num_feats}')

        ## Initialize the weight
        if self.w is None:
            self.w = kwargs.get('init_w', np.random.rand(1, self.num_feats+1))

        ## Initialize learning rate
        if self.lr is None:
            self.lr = kwargs.get('learning_rate', 1)

        ## Initial weight
        #print(f'init_w: {self.w}')

        ## Number of taining dataset
        self.num_train_ds  = len(train_ds)
        self.random_idexes = [i for i in range(self.num_train_ds)]

        ## print sample train datapoint
        #train_dp = train_ds[3]
        #print(f'train_dp:{train_dp}')

        ## Iterate over num_epochs
        for t in range(1, self.num_epochs+1):
            print(f'Trainning epoch {t}/{self.num_epochs}')

            ## Random shuffle the datapoints
            random.shuffle(self.random_idexes)

            for idx in self.random_idexes:
                ## Print current weight
                #print(f'w:{self.w}\nshape:{self.w.shape}')

                ## Get the feats, y
                feats = train_ds[idx][:-1]
                y     = train_ds[idx][-1]

                ## Extend the feats with 1
                x = np.hstack((feats, [1]))[np.newaxis, :]

                ## Print x, y
                #print(f'x:{x}\nshape:{x.shape}')
                #print(f'y:{y}')

                ## Perform dot product
                wx = np.dot(x, self.w.transpose()) 

                ## Print wx
                #print(f'wx:{wx}\nshape{wx.shape}')

                ## Update the weight
                if y * wx[0][0] <= 0:
                    self.w += self.lr * x * y
                    
                    ## Print updated weight
                    #print(f'w:{self.w}\nshape:{self.w.shape}')

        ## Return the weight
        return self.w

    def test_datapoint(self, datapoint):
        '''
        Test a datapoint
        '''
        ## Get the feats
        feats = datapoint[:-1]

        ## Extend the feats with 1
        x = np.hstack((feats, [1]))[np.newaxis, :]

        ## Perform dot product
        wx = np.dot(x, self.w.transpose()) 

        return 1 if wx >= 0 else -1


    def test(self, test_ds):
        '''
        Test the dataset
        '''
        predictions = []

        ## Iterate over all the datapoints
        for datapoint in test_ds:
            ## Predict on the datapoint            
            pred = self.test_datapoint(datapoint=datapoint)

            ## Add to the prediction
            predictions.append(np.hstack((datapoint[:-1], pred)))

        return np.array(predictions)

class Voted_Perceptron(Standard_Perceptron):
    '''
    Class for voted perceptron
    '''

    def __init__(self, num_epochs=10, *args, **kwargs) -> None:
        '''
        Initialize the voted perceptron with default values
        '''
        super().__init__(num_epochs=num_epochs, args=args, kwargs=kwargs)

    def train(self, train_ds, *args, **kwargs):
        '''
        Train the average perceptron
        '''

        ## Get the number of features
        self.num_feats = len(train_ds[0]) - 1
        #print(f'num_feats: {num_feats}')

        ## Initialize the weight
        if self.w is None:
            self.w = kwargs.get('init_w', np.random.rand(1, self.num_feats+1))

        ## Initialize learning rate
        if self.lr is None:
            self.lr = kwargs.get('learning_rate', 1)

        ## Enable random shuffle
        random_shuffle = kwargs.get('random_shuffle', False)

        ## Initial weight
        #print(f'init_w: {self.w}')

        ## Number of taining dataset
        self.num_train_ds  = len(train_ds)
        self.random_idexes = [i for i in range(self.num_train_ds)]

        ## print sample train datapoint
        #train_dp = train_ds[3]
        #print(f'train_dp:{train_dp}')

        self.weight_count_dic = {0: {'w': copy.deepcopy(self.w), 'c': 0}}
        m = 0
        ## Iterate over num_epochs
        for t in range(1, self.num_epochs+1):
            print(f'Trainning epoch {t}/{self.num_epochs}')

            ## Random shuffle the datapoints
            if random_shuffle:
                random.shuffle(self.random_idexes)

            for idx in self.random_idexes:

                ## Get the feats, y
                feats = train_ds[idx][:-1]
                y     = train_ds[idx][-1]

                ## Extend the feats with 1
                x = np.hstack((feats, [1]))[np.newaxis, :]

                ## Print x, y
                #print(f'x:{x}\nshape:{x.shape}')
                #print(f'y:{y}')

                ## Perform dot product
                wx = np.dot(x, self.w.transpose()) 

                ## Print wx
                #print(f'wx:{wx}\nshape{wx.shape}')

                ## Update the weight
                if y * wx[0][0] <= 0:
                    ## Print x, y
                    #print(f'x:{x}\nshape:{x.shape}')
                    #print(f'y:{y}')
                    ## Print current weight
                    #print(f'w:{self.w}\nshape:{self.w.shape}')
                    self.w += self.lr * x * y
                    
                    ## Print updated weight
                    #print(f'w:{self.w}\nshape:{self.w.shape}')
                    m += 1
                    self.weight_count_dic[m] = {'w': copy.deepcopy(self.w), 'c': 1}
                    #print(f'self.weight_count_dic[{m}]:{self.weight_count_dic[m]}')
                    #input()
                else:
                    self.weight_count_dic[m]['c'] += 1

                #print(f'self.weight_count_dic:{self.weight_count_dic}')
                #input()

        ## Return the weight
        return self.weight_count_dic

    def test_datapoint(self, datapoint):
        '''
        Test a datapoint
        '''
        ## Get the feats
        feats = datapoint[:-1]

        ## Extend the feats with 1
        x = np.hstack((feats, [1]))[np.newaxis, :]

        ## Perform dot product
        sum_c_sgn_wx = 0
        for _, w_c in self.weight_count_dic.items():
            w = w_c['w']
            c = w_c['c']
            wx = np.dot(x, w.transpose()) 
            sgn_wx = 1 if wx >= 0 else -1
            c_sgn_wx = sgn_wx * c
            sum_c_sgn_wx += c_sgn_wx

        return 1 if sum_c_sgn_wx >= 0 else -1


class Average_Perceptron(Standard_Perceptron):
    '''
    Class for average perceptron
    '''

    def __init__(self, num_epochs=10, *args, **kwargs) -> None:
        '''
        Initialize the average perceptron with default values
        '''
        super().__init__(num_epochs=num_epochs, args=args, kwargs=kwargs)

    def train(self, train_ds, *args, **kwargs):
        '''
        Train the average perceptron
        '''

        ## Get the number of features
        self.num_feats = len(train_ds[0]) - 1
        #print(f'num_feats: {num_feats}')

        ## Initialize the weight
        if self.w is None:
            self.w = kwargs.get('init_w', np.random.rand(1, self.num_feats+1))

        ## Initialize learning rate
        if self.lr is None:
            self.lr = kwargs.get('learning_rate', 1)

        ## Enable random shuffle
        random_shuffle = kwargs.get('random_shuffle', False)

        ## Initial weight
        #print(f'init_w: {self.w}')

        ## Number of taining dataset
        self.num_train_ds  = len(train_ds)
        self.random_idexes = [i for i in range(self.num_train_ds)]

        ## print sample train datapoint
        #train_dp = train_ds[3]
        #print(f'train_dp:{train_dp}')

        ## Iterate over num_epochs
        self.a_w = copy.deepcopy(self.w)
        for t in range(1, self.num_epochs+1):
            print(f'Trainning epoch {t}/{self.num_epochs}')

            ## Random shuffle the datapoints
            if random_shuffle:
                random.shuffle(self.random_idexes)


            for idx in self.random_idexes:

                ## Get the feats, y
                feats = train_ds[idx][:-1]
                y     = train_ds[idx][-1]

                ## Extend the feats with 1
                x = np.hstack((feats, [1]))[np.newaxis, :]

                ## Print x, y
                #print(f'x:{x}\nshape:{x.shape}')
                #print(f'y:{y}')

                ## Perform dot product
                wx = np.dot(x, self.w.transpose()) 

                ## Print wx
                #print(f'wx:{wx}\nshape{wx.shape}')

                ## Update the weight
                if y * wx[0][0] <= 0:
                    self.w += self.lr * x * y
                    
                self.a_w += self.w

        ## Return the average weight
        return self.a_w

    def test_datapoint(self, datapoint):
        '''
        Test a datapoint
        '''
        ## Get the feats
        feats = datapoint[:-1]

        ## Extend the feats with 1
        x = np.hstack((feats, [1]))[np.newaxis, :]

        ## Perform dot product
        wx = np.dot(x, self.a_w.transpose()) 

        return 1 if wx >= 0 else -1

