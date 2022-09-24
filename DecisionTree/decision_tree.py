## External library import
import numpy as np

from common_functions import *


## Node Class
class Tree_Node(object):
    '''
    Node class for the ID3 algorithm
    '''
    def __init__(self, node_name='root', node_val=None, node_median_val=None, attrib_name_col_dic=None, attrib_name_vals_dic=None, leaf_node=False, node_label=None, depth=0) -> None:
        '''
        Initializes the Tree_Node with default values
        '''
        self.node_name            = node_name
        self.node_val             = node_val
        self.node_median_val      = node_median_val
        self.attrib_name_col_dic  = attrib_name_col_dic
        self.attrib_name_vals_dic = attrib_name_vals_dic
        self.children             = list()
        self.leaf_node            = leaf_node
        self.node_label           = node_label
        self.depth                = depth
        
    def set_node_name(self, node_name):
        '''
        Sets the node name
        '''
        self.node_name = node_name

    def get_node_name(self):
        '''
        Returns node name
        '''
        return self.node_name

    def set_node_val(self, node_val):
        '''
        Sets the node value
        '''
        self.node_val = node_val

    def get_node_val(self):
        '''
        Returns node value
        '''
        return self.node_val

    def set_node_median_val(self, node_median_val):
        '''
        Sets the node median value
        '''
        self.node_median_val = node_median_val

    def get_node_median_val(self):
        '''
        Returns node median value
        '''
        return self.node_median_val

    def set_depth(self, depth):
        '''
        Sets the depth of the Node
        '''
        self.depth = depth

    def get_depth(self):
        '''
        Gets the depth of the Node
        '''
        return self.depth

    def set_leaf_node(self, leaf_node):
        '''
        Sets the node as a leaf node
        '''
        self.leaf_node = leaf_node

    def is_leaf_node(self):
        '''
        Returns True if the node is a leaf node
        '''
        return self.leaf_node

    def set_node_label(self, node_label):
        '''
        Sets the node as a leaf node
        '''
        self.node_label = node_label

    def get_node_label(self):
        '''
        Returns node label
        '''
        return self.leaf_node

class Decision_Tree(object):
    '''
    Class for the decision tree
    '''
    def __init__(self, max_depth=10, entropy_func='entropy'):
        '''
        Initialize the Decision Tree
        '''
        self.max_depth    = max_depth
        self.entropy_func = entropy_func
        self.dtroot       = Tree_Node()
        self.train_ds     = None
        self.num_def_val  = "Numeric"
        self.numeric_col_list = None

        ## Internal attributes
        self.num_features         = None


    def set_max_depth(self, max_depth):
        '''
        Sets the maximum depth
        '''
        self.max_depth = max_depth

    def get_max_depth(self):
        '''
        Returns maximum depth of the tree
        '''
        return self.max_depth

    def set_max_depth(self, max_depth):
        '''
        Sets the maximum depth
        '''
        self.max_depth = max_depth

    def get_max_depth(self):
        '''
        Returns maximum depth of the tree
        '''
        return self.max_depth

    def id3_algorithm(self, dataset, attrib_name_col_dic, attrib_name_vals_dic, depth=0):
        '''
        ID3 Algorithm: Takes following inputs --
            Inputs:
            1. dataset i.e. set of labeled training example
            2. attrib_name_col_dic: {'attrib_1':0, 'attrib_2':1 ..., 'label':n}
            3. attrib_name_vals_dic: {'attrib_1':['val1', 'val2'], 'attrib_2':['high', 'low'], ...}
            4. Current depth of the node
            Output
            Modified dt_root_node
        '''
        ## Feature label split
        features = dataset[:,:-1]
        labels   = dataset[:,-1]

        ## Create a node
        dt_root_node = Tree_Node()
        ## Set passed attributes to the dt_root_node
        dt_root_node.depth = depth
        dt_root_node.attrib_name_col_dic = attrib_name_col_dic
        dt_root_node.attrib_name_vals_dic = attrib_name_vals_dic
        #### Basecases:
        ## Basecase: Max depth has reached:
        # 2. Mark the node as leaf node
        # 3. Set the node label as the majority label
        if dt_root_node.depth >= self.max_depth:
            dt_root_node.set_node_name('leaf_node')
            dt_root_node.set_node_val('max_depth')
            dt_root_node.children   = list()
            dt_root_node.set_leaf_node(True)
            majority_label = get_majority_label(labels=labels)
            dt_root_node.set_node_label(majority_label)
            return dt_root_node

        ## Basecase: All the examples have same label
        # 1. Mark the node as leaf node
        # 2. Set the node label as the majority label
        # 3. Return the modified dt_root_node
        if are_all_labels_same(labels=labels):
            dt_root_node.set_node_name('leaf_node')
            dt_root_node.set_node_val('pure_labels')
            dt_root_node.children   = list()
            dt_root_node.set_leaf_node(True)
            majority_label = labels[0]
            dt_root_node.set_node_label(majority_label)
            return dt_root_node

        ## Basecase: No attributes are left to split upon
        # 1. Mark the node as leaf node
        # 2. Set the node label as the majority label
        # 3. Return the modified dt_root_node
        if len(attrib_name_vals_dic) == 0:
            dt_root_node.set_node_name('leaf_node')
            dt_root_node.node_val   = 'no_attrib_left'
            dt_root_node.children   = list()
            dt_root_node.set_leaf_node(True)
            majority_label = get_majority_label(labels=labels)
            dt_root_node.set_node_label(majority_label)
            return dt_root_node
        
        #### Otherwise: Non-base cases 
        ## Get the attribute to split on
        max_attrib, max_gain = get_attribute_for_max_info_gain(dataset=dataset, attrib_name_vals_dic=attrib_name_vals_dic, attrib_name_col_dic=attrib_name_col_dic, return_pair=True, entropy_func=self.entropy_func) 
        ## Initialize sub dataset
        sub_dataset = {}
        ## Get the new set of attribute and values dictionary
        new_attrib_name_vals_dic = {attrib:attrib_vals for attrib, attrib_vals in attrib_name_vals_dic.items() if attrib != max_attrib}
        
        ## Iterate for each attrib value
        numeric_case = False
        if isinstance(attrib_name_vals_dic[max_attrib], str) and attrib_name_vals_dic[max_attrib] == self.num_def_val:
            numeric_case = True
            attrib_vals = ['left', 'right']
        else:
            attrib_vals = attrib_name_vals_dic[max_attrib]
        for val in attrib_vals:
            ## Split the dataset based on attribute value
            if numeric_case:
                sub_dataset[val] = split_based_on_attrib(attrib=max_attrib, val=self.num_def_val, attrib_name_col_dic=attrib_name_col_dic, dataset=dataset, return_dataset=True, num_def_val=self.num_def_val, first_half=val=='left')
            else:
                sub_dataset[val] = split_based_on_attrib(attrib=max_attrib, val=val, attrib_name_col_dic=attrib_name_col_dic, dataset=dataset, return_dataset=True)
            
            child_node = Tree_Node()
            ## If we have some datapoint in the splitted dataset on the attrib value
            if len(sub_dataset[val]) > 0:
                child_node = self.id3_algorithm(dataset=sub_dataset[val], attrib_name_col_dic=attrib_name_col_dic, attrib_name_vals_dic=new_attrib_name_vals_dic, depth=depth+1)

            ## If we do not have any datapoint in the splitted dataset on the attrib value
            else:
                #print(f'Dataset is empty returned label: {stats.mode(dataset[:,-1])[0][0]}')
                child_node.set_node_name('leaf_node')
                child_node.set_node_val('no_attrib_left')
                child_node.children = list()
                child_node.set_leaf_node(True)
                majority_label = get_majority_label(dataset[:,-1])
                child_node.set_node_label(majority_label)
                child_node.set_depth(depth+1)
                child_node.attrib_name_col_dic  = attrib_name_col_dic
                child_node.attrib_name_vals_dic = new_attrib_name_vals_dic

            ## Modify some attributes of the child node
            child_node.set_node_name(max_attrib)
            child_node.set_node_val(val)
            if numeric_case:
                median_val = median_val = median(map(int,dataset[:,attrib_name_col_dic[max_attrib]]))
                child_node.set_node_median_val(median_val)

            ## Append the child nodes to the parent nodes
            dt_root_node.children.append(child_node)

        ## Return the root node
        return dt_root_node

    def build_tree(self, dataset, attrib_name_col_dic, attrib_name_vals_dic, max_depth=None, entropy_func=None):
        '''
        Build the decesion tree
        '''
        
        ## Add the training dataset to the class
        self.train_ds = dataset

        ## Set the max number of features
        self.num_features = len(dataset[0]) - 1

        ## Update max dataset if it is specified
        if max_depth is not None:
            self.max_depth = max_depth

        ## Update entropy_func if it is specified
        if entropy_func is not None:
            self.entropy_func = entropy_func

        #### Logic for building the decision tree
        self.dtroot = self.id3_algorithm(dataset=dataset, attrib_name_col_dic=attrib_name_col_dic, attrib_name_vals_dic=attrib_name_vals_dic, depth=0)

        return self.dtroot

    def print_tree_dfs(self, node=None, prefix=None):
        if node:
            root_node = node
            prefix    = ":".join([prefix,str(root_node.node_name)+'_'+str(root_node.node_val)])
        else:
            root_node = self.dtroot
            prefix    = "_".join([str(root_node.node_name), str(root_node.node_val)])

        if root_node.children:
            for child_node in root_node.children:
                self.print_tree_dfs(node=child_node, prefix=prefix)
        else:
            print(f'{prefix}:{root_node.node_label}')
        

    def print_tree(self, dfs=False):
        '''
        Prints the Tree
        '''
        if dfs:
            return self.print_tree_dfs()

        stack = []
        stack.append(self.dtroot)

        while stack:
            root_node = stack.pop(0)
            if root_node is not None:
                print(f'Depth:{root_node.depth}, Node/Atrribute:: Name:{root_node.node_name}, Value:{root_node.node_val}, leaf_node:{root_node.leaf_node}, label:{root_node.node_label}')
                for child_node in root_node.children:
                    if child_node is not None:
                        stack.append(child_node)
            

    def predict_datapoint(self, datapoint):
        '''
        Predict on test datapoint
        '''
        dtroot = self.dtroot

        while not dtroot.leaf_node:
            for child_node in dtroot.children:
                if child_node:
                    node_attrib     = child_node.node_name
                    node_val        = child_node.node_val
                    node_attrib_col = child_node.attrib_name_col_dic[node_attrib]
                    if datapoint[node_attrib_col] == node_val:
                        dtroot = child_node
                        break

        return dtroot.node_label

    def predict_dataset(self, dataset):
        '''
        Predict on test dataset
        '''
        if len(dataset[0]) < self.num_features:
            print(f'Test dataset does not have all the features')
            return False

        test_ds = dataset[:,:self.num_features]
        predictions = []
        for i, test_datapoint in enumerate(test_ds):
            #print(f'Predicting datapoint:{i+1}/{len(test_ds)}')
            prediction = self.predict_datapoint(test_datapoint)
            test_datapoint = np.hstack((test_datapoint, [prediction]))
            predictions.append(test_datapoint)

        return np.array(predictions)

    ## Sets the numeric column list
    def set_numeric_col_list(self, numeric_col_list):
        self.numeric_col_list = numeric_col_list

    ## Gets the numeric column list
    def get_numeric_col_list(self):
        return self.numeric_col_list


    '''
    ## Stores the median values
    def store_median_vals(self, dataset, col_attrib_name_dic):
        train_ds_median = {}
        for col in self.numeric_col_list:
            train_ds_median[]
    '''



    def predict_datapoint_numeric(self, datapoint):
        '''
        Predict on test datapoint
        '''
        dtroot = self.dtroot

        while not dtroot.leaf_node:
            for child_node in dtroot.children:
                if child_node:
                    node_attrib     = child_node.node_name
                    node_val        = child_node.node_val
                    node_attrib_col = child_node.attrib_name_col_dic[node_attrib]

                    if node_attrib_col in self.numeric_col_list:
                        if int(datapoint[node_attrib_col]) <= child_node.node_median_val and node_val == 'left':
                            dtroot = child_node
                            break
                        if int(datapoint[node_attrib_col]) > child_node.node_median_val and node_val == 'right':
                            dtroot = child_node
                            break
                    else:
                        if datapoint[node_attrib_col] == node_val:
                            dtroot = child_node
                            break

        return dtroot.node_label

    def predict_dataset_numeric(self, dataset):
        '''
        Predict on test dataset
        '''
        if len(dataset[0]) < self.num_features:
            print(f'Test dataset does not have all the features')
            return False

        test_ds = dataset[:,:self.num_features]
        predictions = []
        for i, test_datapoint in enumerate(test_ds):
            #print(f'Predicting datapoint:{i+1}/{len(test_ds)}')
            prediction = self.predict_datapoint_numeric(test_datapoint)
            test_datapoint = np.hstack((test_datapoint, [prediction]))
            predictions.append(test_datapoint)

        return np.array(predictions)