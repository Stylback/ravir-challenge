import matplotlib.pyplot as plt
import numpy as np

def plot(size_x, size_y, title, x_label, y_label, legend, print_keys, model_hist):
    
    keys = list(model_hist.history.keys())
    if print_keys:
        print("The keys are: ", keys)

    plt.figure(figsize=(size_x, size_y))
    plt.title(title)
    for key in keys:
        plt.plot(model_hist.history[key], label=key)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    if legend:
        plt.legend();
    
    plt.grid(linestyle = '--', linewidth = 0.5)
    plt.show()
