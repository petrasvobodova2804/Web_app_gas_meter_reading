import csv
import statistics
import matplotlib.pyplot as plt

def do_statistics(gasmeter_values):

        # Solves the reset to 0000 of gasmeter
        for i in range(0, len(gasmeter_values)-1):
            reset_index = None
            if (gasmeter_values[i] > gasmeter_values[i+1]):
                reset_index = i
                print("Gasmeter was reset")
                if reset_index is not None:
                    for j in range(reset_index+1, len(gasmeter_values)):
                        gasmeter_values[j] = gasmeter_values[j] + gasmeter_values[reset_index]

        # Compute differences between weekly values
        diff_values = []
        for i in range(0, len(gasmeter_values)-1):
            res_ = gasmeter_values[i+1] - gasmeter_values[i]
            diff_values.append(res_)

        # Calculate statistics
        mean = statistics.mean(gasmeter_values)
        minimum = min(gasmeter_values)
        maximum = max(gasmeter_values)
        standard_deviation = statistics.stdev(gasmeter_values)

        # Print the statistics
        #print("Mean:", mean)
        #print("Minimum:", minimum)
        #print("Maximum:", maximum)
        #print("Standard Deviation:", standard_deviation)

        # Plot the gas meter readings
        week_number = len(gasmeter_values)
        week_numbers = []
        for k in range(0,week_number):
             week_numbers.append(int(k+1))
             
        plt.plot(week_numbers, gasmeter_values)
        plt.xlabel('Week Number')
        plt.ylabel('Gas Meter Reading')
        plt.title('Gas Meter Readings Over Time')
        plt.savefig("static/uploads/gas_meter.png")
        plt.close()


        # Plot the gas meter readings
        plt.plot(diff_values)
        plt.xlabel('Week Number')
        plt.ylabel('Differencies')
        plt.title('Differencies of kWh across weeks')
        plt.savefig("static/uploads/gas_meter_diff.png")
        plt.close()

    

