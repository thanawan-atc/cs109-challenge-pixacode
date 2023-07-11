import math
import csv
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# Correct Pixacode (1, 8, 6, 2)
USER_CLICK_DATA_T= [(243, 1694), (640, 2131), (916, 1927), (643, 1763)]
PASSCODE_DATA_T = [[(247, 1723), (637, 2143), (941, 1924), (651, 1793)],
                 [(241, 1693), (665, 2152), (924, 1933), (671, 1779)],
                 [(250, 1079), (643, 2142), (905, 1943), (683, 1775)],
                 [(244, 1715), (629, 2159), (921, 1949), (694, 1747)],
                 ]

# Incorrect Pixacode
USER_CLICK_DATA_F = [(243, 1694), (640, 2131), (916, 1927), (450, 1775)]
PASSCODE_DATA_F = [[(243, 1694), (640, 2131), (916, 1927), (650, 1775)],
                 [(243, 1694), (640, 2131), (916, 1927), (650, 1775)],
                 [(243, 1694), (640, 2131), (916, 1927), (650, 1775)],
                 [(243, 1694), (640, 2131), (916, 1927), (650, 1775)],
                 ]

PASSCODE_LENGTH = 4

SCREEN_WIDTH = 1170     # Width of I-phone 12 Pro
SCREEN_HEIGHT = 2532    # Height of I-phone 12 Pro

BENCHMARK = 100     # Benchmark for Likelihood(K)


def calculate_stats_clicks_data_i(filename, samples_x, samples_y):
    """
    Get sample data from clicks_data_i.csv and put x-diff and y-diff into the lists
    """
    x_coordinates = []
    y_coordinates = []

    reader = csv.DictReader(open(filename))
    for row in reader:
        x_coordinates.append(int(row['x']))
        y_coordinates.append(int(row['y']))
    mean_x = np.mean(x_coordinates)
    mean_y = np.mean(y_coordinates)
    diff_x = [x_coordinate - mean_x for x_coordinate in x_coordinates]
    diff_y = [y_coordinate - mean_y for y_coordinate in y_coordinates]

    samples_x.extend(diff_x)    # samples_x.append(diff_x) for plotting 1-9 graphs
    samples_y.extend(diff_y)    # samples_y.append(diff_y) for plotting 1-9 graphs


def bootstrap_standard_error(samples_diff):
    """
    Find standard error of sample variance by using bootstrapping
    """
    resample_vars = []
    for i in range(10000):
        resamples_diff = np.random.choice(samples_diff, len(samples_diff), replace=True)
        var_resample_diff = np.var(resamples_diff) * len(resamples_diff) / (len(resamples_diff) - 1)
        resample_vars.append(var_resample_diff)
    standard_error_var = math.sqrt(np.var(resample_vars) * len(resample_vars) / (len(resample_vars) - 1)
                                        / len(resample_vars))
    return standard_error_var


def calculate_stats_clicks_data():
    """
    Find sample variance X and sample variance Y from data
    """
    samples_diff_x = []
    samples_diff_y = []
    calculate_stats_clicks_data_i('clicks_data_1.csv', samples_diff_x, samples_diff_y)
    calculate_stats_clicks_data_i('clicks_data_2.csv', samples_diff_x, samples_diff_y)
    calculate_stats_clicks_data_i('clicks_data_3.csv', samples_diff_x, samples_diff_y)
    calculate_stats_clicks_data_i('clicks_data_4.csv', samples_diff_x, samples_diff_y)
    calculate_stats_clicks_data_i('clicks_data_5.csv', samples_diff_x, samples_diff_y)
    calculate_stats_clicks_data_i('clicks_data_6.csv', samples_diff_x, samples_diff_y)
    calculate_stats_clicks_data_i('clicks_data_7.csv', samples_diff_x, samples_diff_y)
    calculate_stats_clicks_data_i('clicks_data_8.csv', samples_diff_x, samples_diff_y)
    calculate_stats_clicks_data_i('clicks_data_9.csv', samples_diff_x, samples_diff_y)

    sample_var_x = np.var(samples_diff_x) * len(samples_diff_x) / (len(samples_diff_x) - 1)
    sample_var_y = np.var(samples_diff_y) * len(samples_diff_y) / (len(samples_diff_y) - 1)
    print('Sample Var X:', sample_var_x)
    print('Sample Var Y:', sample_var_y)

    #sns.displot(samples_diff_x, kind="kde")
    #plt.show()

    #sns.displot(samples_diff_y, kind="kde")
    #plt.show()

    standard_error_var_x = bootstrap_standard_error(samples_diff_x)
    standard_error_var_y = bootstrap_standard_error(samples_diff_y)
    print('Standard Error Sample Var X:', standard_error_var_x)
    print('Standard Error Sample Var Y:', standard_error_var_y)
    return (sample_var_x, sample_var_y)


def pdf_key_given_user(coordinate_key_input, coordinate_key_passcode, sample_var):
    """
    Compute probability density function of ith key input given the person is a real user
    """
    pdf_x_given_user = stats.norm.pdf(coordinate_key_input[0], coordinate_key_passcode[0], math.sqrt(sample_var[0]))
    pdf_y_given_user = stats.norm.pdf(coordinate_key_input[1], coordinate_key_passcode[1], math.sqrt(sample_var[1]))
    joint_pdf_given_user = pdf_x_given_user * pdf_y_given_user
    return joint_pdf_given_user


def pdf_input_given_user(input, passcode):
    """
    Compute probability density function of the Pixacode input given the person is a real user
    """
    sample_var = calculate_stats_clicks_data()

    total_pdf_given_user = 1
    for i in range(len(input)):
        total_pdf_given_user *= pdf_key_given_user(input[i], passcode[i], sample_var)
    print("pdf_input_given_user:", total_pdf_given_user)
    return total_pdf_given_user


def pdf_key_given_not_user(coordinate_key_input):
    """
    Compute probability density function of ith key input given the person is a fake user
    """
    pdf_x_given_not_user = stats.norm.pdf(coordinate_key_input[0], SCREEN_WIDTH * 1 / 2, SCREEN_WIDTH * 1 / 3)
    pdf_y_given_not_user = stats.norm.pdf(coordinate_key_input[1], SCREEN_HEIGHT * 2 / 3, SCREEN_HEIGHT * 1 / 3)
    joint_pdf_given_not_user = pdf_x_given_not_user * pdf_y_given_not_user
    return joint_pdf_given_not_user


def pdf_input_given_not_user(input):
    """
    Compute probability density function of the Pixacode input given the person is a fake user
    """
    total_pdf_given_not_user = 1
    for i in range(len(input)):
        total_pdf_given_not_user *= pdf_key_given_not_user(input[i])
    print('pdf_input_given_not_user:', total_pdf_given_not_user)
    return total_pdf_given_not_user


def expected_set_passcode(set_passcodes):
    """
    Find the expected Pixacode
    """
    expected_passcode = []
    for i in range(PASSCODE_LENGTH):
        set_passcodes_i = [row[i] for row in set_passcodes]
        set_passcodes_i_x = [coordinate_i[0] for coordinate_i in set_passcodes_i]
        set_passcodes_i_y = [coordinate_i[1] for coordinate_i in set_passcodes_i]
        expected_passcode_i_x = np.mean(set_passcodes_i_x)
        expected_passcode_i_y = np.mean(set_passcodes_i_y)
        expected_passcode.append((expected_passcode_i_x, expected_passcode_i_y))
    return expected_passcode


def main():
    np.random.seed(seed=1)

    # Set Pixacode
    set_passcodes = PASSCODE_DATA_T
    passcode = expected_set_passcode(set_passcodes)

    # Enter Pixacode and compute the likelihood
    input_passcode = USER_CLICK_DATA_T
    likelihood_real_user = pdf_input_given_user(input_passcode, passcode) / pdf_input_given_not_user(input_passcode)
    print('Likelihood(K):', likelihood_real_user)

    # Determine whether the person is a real or fake user.
    if likelihood_real_user > BENCHMARK:
        print('You entered the right Pixacode. Welcome back!')
    else:
        print('The Pixacode you entered is incorrect. You\'re not the real user')


if __name__ == '__main__':
    main()