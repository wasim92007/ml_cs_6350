## External library import
import matplotlib.pyplot as plt

## Internal library import
from common_functions import *

## Plot single data
def plot_one_data_func(data_1, graph_name, label_dic, *args, **kwargs):
    ## Create the graph directory if it does not exist
    dir_path = os.path.dirname(os.path.realpath(graph_name))
    create_dir(dir_name=dir_path)
    
    ## Creates x if not provided
    it = kwargs.get('it', list(range(len(data_1))))
    
    ## Graph related settings
    graph = plt.figure()    ## Wasim
    
    plt.plot(it, data_1, 'r', linewidth=1.0)
    
    plt.grid(visible=True, color='#f8b8ff', linestyle='-.',)  ## Wasim
    plt.tick_params(axis='both', direction='out', length=6, width=1, labelcolor='b', colors='r', grid_color='gray', grid_alpha=0.1) ## Wasim
    
    plt.xlabel(label_dic['xlabel'],  fontsize=20)
    plt.ylabel(label_dic['ylabel'],  fontsize=20)
    plt.legend(label_dic['legend'], bbox_to_anchor=(1.05, 1.0),loc=2)
    plt.title(label_dic['title'])
    graph.savefig(graph_name, dpi=300, bbox_inches='tight', facecolor="#efffed")
    plt.close()

## Plot two data
def plot_two_data_func(data_1, data_2, graph_name, label_dic, *args, **kwargs):
    ## Create the graph directory if it does not exist
    dir_path = os.path.dirname(os.path.realpath(graph_name))
    create_dir(dir_name=dir_path)
    
    ## Creates x if not provided
    it = kwargs.get('it', list(range(len(data_1))))
    
    ## Graph related settings
    graph = plt.figure()    ## Wasim
    
    plt.plot(it, data_1, 'r', linewidth=1.0)
    plt.plot(it, data_2, 'b', linewidth=1.0)
    
    plt.grid(visible=True, color='#f8b8ff', linestyle='-.',)  ## Wasim
    plt.tick_params(axis='both', direction='out', length=6, width=1, labelcolor='b', colors='r', grid_color='gray', grid_alpha=0.1) ## Wasim
    
    plt.xlabel(label_dic['xlabel'],  fontsize=20)
    plt.ylabel(label_dic['ylabel'],  fontsize=20)
    plt.legend(label_dic['legend'], bbox_to_anchor=(1.05, 1.0),loc=2)
    plt.title(label_dic['title'])
    graph.savefig(graph_name, dpi=300, bbox_inches='tight', facecolor="#efffed")
    plt.close()