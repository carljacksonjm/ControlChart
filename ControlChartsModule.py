#%% Functions
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


class NelsonTests:
    def __init__(self, data, data_mean, data_sigma):
        self.data = data
        self.data_mean = data_mean
        self.data_sigma = data_sigma
        self.UCL = self.data_mean + 3*self.data_sigma
        self.LCL = self.data_mean - 3*self.data_sigma
    
    def failed_test(array_data, test_no):
        """Prints data which have failed the tests
        """
        
        if len(array_data)>0:            
            print(f"Test {test_no} failed\nAlarm at ", end = "")
            for bad_point in array_data[:-1,0]:
                print(f"{bad_point: .2f}", end = ", ")
            print(f"{array_data[-1,0]: .2f}", end = ".\n")
        else:
            print(f"Test {test_no} passed.")
        print("-----")
    def find_array_index(arr_all, arr_subset):
        """Finds positions of a subset array in an arr_all.
        Returns the index of the elements in the subset array that are in the total array
        """
        return [arr_all.tolist().index(x) for x in arr_subset]
        
    def test_1(self):
        """Tests to see if one point is more than 3 standard deviations from the mean
        """
        print(f"Applying Test 1: One point at 3 std dev or more...")
        
        bad_data = self.data[(self.data <= self.LCL) | (self.data >= self.UCL)]            
        bad_data_index = NelsonTests.find_array_index(self.data, bad_data)  
        bad_results = np.stack((bad_data, bad_data_index)).T

        NelsonTests.failed_test(bad_results, 1)
        return bad_results
    
    def test_2(self):
        """
        Nine points in a row on the same side of the center line
        """
        print(f"Applying Test 2: Nine points in a row on the same side of the mean...")
        n_points = 9

        bad_data, bad_data_index = [], []
        #print(data_sample)

        for i in range(n_points, (len(self.data))+1): # iterate from 4th point to the end
            array_index = i-n_points
            suspect_data = self.data[array_index:i] # 0, 9
            l_suspect = len(suspect_data)

            l_bad = max([len(suspect_data[np.where(suspect_data > self.data_mean)]), len(suspect_data[np.where(suspect_data < self.data_mean)])])
            # print(l_bad)

            if l_suspect == l_bad:
                # print("failed test 2\n---")
                # print(type(np.where(suspect_data> m)))

                bad_data.extend(suspect_data[[0]])
                bad_data_index.append(array_index)
                
        bad_data = np.array(bad_data)
        bad_data_index = np.array(bad_data_index)
        bad_results = np.stack((bad_data, bad_data_index)).T

        NelsonTests.failed_test(bad_results, 2)

        return bad_results
    
    def test_3(self):
        """Six points in a row steadily increasing or steadily decreasing

        drift in the process mean
        e.g. Tool wear, deterioration, skill improvement
        """
        print(f"Applying Test 3: Six points trending in the same direction...")
        def check_trend_direction(i2, i1):
            """
            Checks direction a pair of numbers are trending. If i2 > i1, returns +1
            """

            t_dir = 1 if i2 > i1 else 0 if i2 == i1 else -1

            return t_dir

        #print(f"Applying test 3")
        n_points = 6
        bad_data, bad_data_index = [], []
        i = 0
        drift_counter = 0
        array_length = len(self.data)
        #print(f"Array length {array_length}") 
        
        running_trend = 0

        while i < (array_length-1):

            new_running_trend = running_trend + check_trend_direction(self.data[i+1], self.data[i])

            #print(f"Element Inspected {self.data[i+1]} (trending {new_running_trend})")
            # increase / decrease check
            if abs(new_running_trend) <= abs(running_trend):
                # not same direction
                running_trend = 0
            
            else:
                # same direction
                running_trend = new_running_trend
                
                if abs(running_trend) == (n_points-1):
                    #print(f"--\ndrift detected at i = {i+1} - {self.data[i+1]}")

                    drift_start_index = i - (n_points-2)
                    #print(f"drift started at i = {drift_start_index} - {self.data[drift_start_index]}")

                    bad_data.append(self.data[drift_start_index]) # starting point
                    bad_data_index.append(drift_start_index)

                    running_trend = 0
                    i = drift_start_index # jump back to the one after the drift starting point
                    #print(f"jump back to i = {i+1}")  
            
            i += 1

        bad_data = np.array(bad_data)
        bad_data_index = np.array(bad_data_index)
        bad_results = np.stack((bad_data, bad_data_index)).T

        NelsonTests.failed_test(bad_results, 3)

        return bad_results

class ControlChart():
    
    def __init__(self, data_sample, p_stddev):
        self.data_sample = data_sample
        self.fig_name = fig_name
        self.p_stddev
        
    def control_limit_range(starting_lim, ending_lim):
        """Creates an array from the starting limit to the end
        [starting_lim, a2, a3, a4, a5, ending_lim]
        """
        return np.linspace(starting_lim, ending_lim, 7)

    def control_limits(m, stddev):
        """Calculates the control limits from the mean and std dev
        [LCL, xbar - 2 std, xbar - 1 std, ..., UCL]
        """
        return ControlChart.control_limit_range((m - 3*stddev), (m + 3*stddev)) #np.linspace((m - 3*stddev), (m + 3*stddev), 7)
    
    def control_limits_mr(mr_av):
        """Calculates the control limits of a moving range chart from the average moving range
        [0, ..., 3.267 * average mv]
        """
        D4 = 3.267 # sample size-specific D4 anti-biasing constant for n=2
        return ControlChart.control_limit_range(0, mr_av * D4)
    
    def i_to_mr(data_sample):
        """Converts data sample arrage to moving range array
        [[raw val, mr val]]
        """        
        i_max = (data_sample).shape[0]       

        mr_sample = np.zeros((i_max, 3))

        mr_sample[:,0] = data_sample[:,0] # row 1 raw data
        mr_sample[:,2] = data_sample[:,1]

        for i in range(1, i_max):
            mr_sample[i,1] = abs(data_sample[i,0] - data_sample[i-1,0]) # row 2 moving range
        # print("mr sample:", mr_sample)
        return mr_sample
    
    def mrmean_to_stddev(mr_average):
        """Converts moving range mean to a sequential deviation
        """
        d2 = 1.128 # n = 2
        return mr_average/d2
        
        
    def plot_graph(x_vals, y_vals, c_lims, x_string, y_string, title_string,
                   signals = None, fig_name = None, y_lim = None, mean_string = r'$\bar{m}$'):
        """Plots a graph of x_vals against y_vals
        Control limits are labelled
        Any signals are marked
        """
        
        fig, ax = plt.subplots(figsize = (10, 6))

        # plotting
        if signals:
            shift_size = 0.3
            shift_dict = {1: [-shift_size, 0],
            2: [-shift_size, +shift_size],
            3: [-shift_size, -shift_size]}

            # n_signals = len(signals)

            for i, tests in enumerate(signals):
                
                signal_array = signals[tests]
                shift_x = shift_dict[i+1][0]
                shift_y = shift_dict[i+1][1]
                
                print(f"Test {tests}: {signal_array.shape[0]} signals")
                signal_markers = ["+", "x", "X","+", "x", "X"]
                signal_colors = ["red","red","red","blue","blue","blue","green","green","green"]
                if signal_array.shape[0] > 0:
                    # print("[1,:] ", signal_array[1,:])
                    ax.scatter(signal_array[:,1] + shift_x, signal_array[:,0] + shift_y,
                               s = 80, marker=signal_markers[i], color = signal_colors[i],
                               label = f"Test {tests}")

        ax.plot(x_vals, y_vals, linestyle='-', marker='.', color='blue', linewidth = 1, markersize = 10, label = "Data")
    
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        
        # create control lines
        left, right = ax.get_xlim()
        zone_list = ["A","B","C","C","B","A"]
        text_distance = 0.3
        
        for i, c_lim in enumerate(c_lims):
            if i == 0:
                ax.axhline(c_lim, color='red')
                ax.text(right + text_distance, c_lim, "LCL = " + str("{:.2f}".format(c_lim)), color='red')

            elif i == 6:
                ax.axhline(c_lim, color='red')
                ax.text(right + text_distance, c_lim, "UCL = " + str("{:.2f}".format(c_lim)), color='red')

            elif i == 3:
                ax.axhline(c_lim, color='green')
                ax.text(right + text_distance, c_lim, mean_string + " = " + str("{:.2f}".format(c_lim)), color='green')

            else:
                ax.axhline(c_lim, color='orange', ls = "--")

        # zone labels
        for i, zone_label in enumerate(zone_list):
            if i < len(c_lims):
                label_position = (c_lims[i]+c_lims[i+1])/2
                ax.text(right + text_distance, label_position, zone_list[i], color='orange')

        ax.set_title(title_string)        
        ax.set(xlabel = x_string, ylabel = y_string)
        
        if y_lim:
            ax.set_ylim(y_lim[0],y_lim[1])

        if fig_name:
            plt.savefig(fig_name+".png", facecolor='w')

        plt.show()
           
    def mr_graph(data_points,
                 x_string, y_string, title_string, fig_name = None):
        """Plots a moving range graph. Converts data points to moving range values and calculates moving range mean and control limits.

        """
        
        mr_points = ControlChart.i_to_mr(data_points)
        mr_hat = np.mean(mr_points[:, 1])
        
        c_limits = ControlChart.control_limits_mr(mr_hat)
        
        ControlChart.plot_graph(mr_points[:, 2], mr_points[:, 1],
                                c_limits, x_string, y_string, title_string, fig_name = fig_name, mean_string = r'$\bar{MR}$') #mr_points[1, :]      
    
    def control_graph(data_points, p_stddev, x_string, y_string, title_string,
                      p_control_tests = None, fig_name = None, y_lim = None):
        """Plots an indiviual point graph.
        
        Converts data points to moving range values and calculates moving range mean and control limits.

        Calculates std dev, mean and control limits. Applies Nelson tests
        """
        # calc mrhat
        mr_data_points = ControlChart.i_to_mr(data_points) # [[raw value, mr value, original index]]
        mrhat = np.mean(mr_data_points[:,1])
        
        # convert to stddev
        p_stddev_nel = ControlChart.mrmean_to_stddev(mrhat) # print(p_stddev_nel)
        
        # convert to clims
        s_mean = np.mean(mr_data_points[:,0])
        c_limits = ControlChart.control_limits(s_mean, p_stddev_nel)
        
        # tests
        def ApplyTests(test_list):
            """
            Iterate through list of test numbers, get test function and apply to data.
            Return dictionary of {test number: [[value index]]}
            """
            
            test_dictionary = {
                1: nelson_class.test_1(),
                2: nelson_class.test_2(),
                3: nelson_class.test_3()
            }
            
            signal_dictionary = {}

            for t in test_list:
                if t in test_dictionary:
                
                    test_func = test_dictionary[t] # nelson_class.test_1()
                    signal_dictionary[t] = test_func # 1:nelson_class.test_1()
                    
            return signal_dictionary
                
        nelson_class = NelsonTests(mr_data_points[:,0], s_mean, p_stddev_nel)        
        signal_dict = ApplyTests(p_control_tests) # dict {test no: [[signal, index]]}
        
        # Nelson Method
        # n_signals_1 = sum([len(test_array) for test_array in signal_dict.values()])
        # # print(f"Total signals = {n_signals_1}")
        # delta_n_signals = n_signals_1
        
#         while delta_n_signals != 0:
            # return indexes of all signals
            
        def get_signal_indexes(signal_dictionary):
            """
            Takes dictionary of {rule number: [[x value, index]]} and returns a set of all indexes
            """
            
            list_of_sets = [set(test_array[:,1].flatten()) for test_array in signal_dict.values()]
            #print(list_of_sets)
            set_of_sets = set().union(*list_of_sets)
            
            # print("Indexes of signals:",set_of_sets) 
            return set_of_sets
                
        bad_indexes = get_signal_indexes(signal_dict) # set of indexes of signals
        # print("bad i", bad_indexes)
        
        def drop_mr_values(good_bad_array, bad_indexes):
            """
            Takes an array and deletes the rows, returns an array with those rows dropped
            """
            # index_vals_arrays = [np.where(good_bad_array[:,1].flatten() == bad_index) for bad_index in bad_indexes]
            # index_vals_arrays = [np.where(good_bad_array[:,1] == bad_index) for bad_index in bad_indexes]
            # print(index_vals_arrays)
            # print(good_bad_array)
            for bad_index in bad_indexes:
                print("bad index:", bad_index)
                # find row of bad value in array
                # print(np.where(good_bad_array[:,1] == bad_index))
                # row_for_delete = int(np.where(good_bad_array[:,1].flatten() == bad_index)[0])

                # delete row of bad value
                # good_bad_array = np.delete(good_bad_array, bad_index, 0)

            return good_bad_array      
        
        mr_data_points_clean = drop_mr_values(mr_data_points, bad_indexes)
        # print(f"dirty array: {mr_data_points}\n---\nclean array: {mr_data_points_clean}")
                      
            # remove indexes from MR
            # recalculate mrhat
            # convert to stddev
            # convert to clims
            # test (nsignals)
            # n_signals_2
            # delta_n_signals = n_signals_2 - n_signals_1
            # print signals removed and effect on CLs

        # plot
        
        ControlChart.plot_graph(mr_data_points[:,2], mr_data_points[:,0], c_limits, x_string, y_string, title_string, 
                                signals = signal_dict, mean_string = r'$\bar{m}$')



# test_data = np.array([34.10000000000001, 
# 33.80000000000001, 
# 34.19999999999999, 
# 34.30000000000001, 
# 33.999999999999986, 
# 33.70000000000002, 
# 34.30000000000001, 
# 33.8, 
# 33.89999999999999, 
# 34.19999999999999, 
# 34.0, 
# 34.39999999999999, 
# 32.400000000000006, 
# 32.499999999999986, 
# 32.900000000000006, 
# 32.599999999999994, 
# 33.2, 
# 33.0, 
# 33.099999999999994, 
# 33.099999999999994, 
# 33.2, 
# 33.500000000000014, 
# 33.8, 
# 34.099999999999994, 
# 34.30000000000001, 
# 32.80000000000001, 
# 34.69999999999999, 
# 34.099999999999994, 
# 34.60000000000001, 
# 34.5, 
# 35.10000000000001, 
# 35.599999999999994, 
# 35.599999999999994, 
# 35.19999999999999, 
# 36.0, 
# 35.79999999999998, 
# 36.10000000000001, 
# 35.69999999999999, 
# 35.400000000000006, 
# 38
# ])

# test_stddev = 0.5 # 0.276
# test_mean = np.mean(test_data)
#nelson_class = NelsonTests(np.array([1,28,27,26,25,24,25,28,-2,200,25,25,26,27,28,29,30,31,32,33,29]), 26, 0.15)
#nelson_class.test_1() # [[val index],[val2 index2]]
