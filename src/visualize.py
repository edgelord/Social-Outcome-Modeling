import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_field(x,data, title, y, label=None):
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel(y)
    return plt.plot(x, data, label=label)

def visualize_history(history, count, error_only=True):
    samples=len(history['costs'])
    scale=count/samples*1.
    x=[n*scale for n in range(samples)]
    costs = history['costs']
    test_error = history['test_error']
    train_error = history['train_error']
    weights = history['weight_mags']
    train,= plt.plot(x,train_error, label='Training set error')
    test,=plot_field(x,test_error, 'Error', 'Percent Error', label='Test set error')
    plt.legend(handles=[test, train])
    if error_only:
        plt.title('Accuracy Predicting the Number of Urban Residents')
        print "Lowest test error: ", np.min(test_error)
        print "Lowest train error: ", np.min(train_error)
        plt.show()
        return

    
    plot_field(costs, 'Cost Function over Iterations', '$J$')
    plt.figure()
    
    plot_field(weights, 'Weight Vector Magnitude', r'$\mid \theta \mid ^2$')
    plt.show()
