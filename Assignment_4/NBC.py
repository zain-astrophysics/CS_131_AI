import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import classification_report


pathname = 'C:\\Users\\zaina\\OneDrive\\Documents\\Artificial Intelligence\\Assignment_4\\'
file = pathname + 'likelihood.txt'
train_file = pathname + 'dataset.txt'
test_file = pathname + 'testing.txt'

likelihood = np.loadtxt(file)
velocity_data = np.loadtxt(train_file)
testing_data = np.loadtxt(test_file)

#  Define binning range
vmin  = np.nanmin(testing_data)
vmax = np.nanmax(testing_data)
num_bins = likelihood.shape[1]
test_labels = ['b', 'b', 'b', 'a', 'a', 'b', 'a', 'a', 'a', 'b']
test_velocity = testing_data
training_labels = ['b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a',  'a', 'a']
model = int(input("Use  (1) Velocity (2) velocity + acceleration (3) or velocity + acceleration + displacement? "))

#  Compute likelihood indices directly using linear scaling 
def velocity_to_index(v):
    #  Scale velocity to bin index range [0 399]
    if np.isnan(v):
        return -1
    scaled = (v - vmin) / (vmax - vmin) * (num_bins - 1)
    index = int(np.clip(np.floor(scaled),0, num_bins - 1)) # ensure index is within bounds
    return index



# Desiging naive Bayesian  Classification 

# Transition probability
transition_Matrix =np.array([
    [0.9, 0.1],   # bird -> [bird, airplane]
    [0.1, 0.9]    # airplane -> [bird, airplane]
])


def naive_bayes_classification(velocity, likelihood, transition_Matrix):
    # num_bins = likelihood.shape[1]
    # vmin = np.nanmin(velocity)
    # vmax = np.nanmax(velocity)

    prior_bird = 0.5
    prior_airplane = 0.5

    posterior_bird = prior_bird
    posterior_airplane = prior_airplane
    predictions = []

    for v in velocity:
        if np.isnan(v):
            predictions.append('b' if posterior_bird > posterior_airplane else 'a')
            continue
    
        # idx = velocity_to_index(v, vmin, vmax, num_bins)
        idx = velocity_to_index(v)
        likelihood_bird = likelihood[0, idx]
        likelihood_airplane = likelihood[1, idx]


        # Updating priors
        update_priors_bird = posterior_bird * transition_Matrix[0,0] + posterior_airplane*transition_Matrix[0,1]   # previous posterior becomes prior
        update_priors_airplane = posterior_airplane * transition_Matrix[1,1] + posterior_bird * transition_Matrix[1,0]

        # Updating posteriors
        posterior_bird = update_priors_bird * likelihood_bird
        posterior_airplane = update_priors_airplane * likelihood_airplane

        total = posterior_airplane + posterior_bird

        if total == 0:
            posterior_bird = 0.5
            posterior_airplane = 0.5
        else:
            posterior_bird /= total
            posterior_airplane /= total

        
        predictions.append('b' if posterior_bird > posterior_airplane else 'a')

    bird_count = predictions.count('b')
    airplane_count = predictions.count('a')
    final_classification = 'b' if bird_count > airplane_count else 'a'
    return predictions, final_classification
results = []
print("Running Velocity feature")

def calculate_acceleration(velocity):
    acceleration_data = []
    for track in velocity:
        acceleration = [np.nan]
        for i in range(1, len(track)):
            if not np.isnan(track[i]) and not np.isnan(track[i-1]):
                acc = (track[i] - track[i-1])
            else:
                acc = np.nan
            acceleration.append(acc)
        acceleration_data.append(acceleration)
    return np.array(acceleration_data)

acceleration_data = calculate_acceleration(velocity_data)
bird_acceleration = np.concatenate(acceleration_data[:10])   # This would us single array a = [a1, a2, ... a600, a1, a2, ....a600], size = 6000
airplane_acceleration = np.concatenate(acceleration_data[10:])

# Calculate likelihood for bird and airplane acceleration
def calculate_likelihood(bird_acceleration, airplane_acceleration):
    bird_acceleration = bird_acceleration[~np.isnan(bird_acceleration)]
    airplane_acceleration = airplane_acceleration[~np.isnan(airplane_acceleration)]

    bird_mu, bird_std = norm.fit(bird_acceleration)
    airplane_mu, airplane_std = norm.fit(airplane_acceleration)

    return bird_mu, bird_std, airplane_mu, airplane_std

bird_mu, bird_std, airplane_mu, airplane_std = calculate_likelihood(bird_acceleration, airplane_acceleration)

# ##### Updated Classifier NBC
def naive_bayes_classification_updated(velocity, acceleration, likelihood, transition_Matrix, bird_mu, bird_std, airplane_mu, airplane_std):

    prior_bird = 0.5
    prior_airplane = 0.5

    posterior_bird = prior_bird
    posterior_airplane = prior_airplane
    predictions = []

    for v, a in zip(velocity, acceleration):
        if np.isnan(v) and np.isnan(a):
            predictions.append('b' if posterior_bird > posterior_airplane else 'a')
            continue
            
        idx = velocity_to_index(v)
        # velocity likelihoods
        v_likelihood_bird = 1 if np.isnan(v) else likelihood[0, idx]
        v_likelihood_airplane = 1 if np.isnan(v) else likelihood[1, idx]

        # Acceleration likelihoods
        a_likelihood_bird = 1 if np.isnan(a) else norm.pdf(a, bird_mu, bird_std)
        a_likelihood_airplane = 1 if np.isnan(a) else norm.pdf(a, airplane_mu, airplane_std)

        # Combined likelihoods
        combined_bird_likelihood = v_likelihood_bird * a_likelihood_bird
        combined_airplane_likelihood = v_likelihood_airplane * a_likelihood_airplane
        # Updating priors
        update_priors_bird = posterior_bird * transition_Matrix[0,0] + posterior_airplane*transition_Matrix[0,1]   # previous posterior becomes prior
        update_priors_airplane = posterior_airplane * transition_Matrix[1,1] + posterior_bird * transition_Matrix[1,0]

        # Updating posteriors
        posterior_bird = update_priors_bird * combined_bird_likelihood
        posterior_airplane = update_priors_airplane * combined_airplane_likelihood

        total = posterior_airplane + posterior_bird

        if total == 0:
            posterior_airplane = 0.5
            posterior_bird = 0.5
        else:
            posterior_bird /= total
            posterior_airplane /= total
        
        predictions.append('b' if posterior_bird > posterior_airplane else 'a')

    bird_count = predictions.count('b')
    airplane_count = predictions.count('a')
    final_classification = 'b' if bird_count > airplane_count else 'a'
    return predictions, final_classification


#  ***** Running Test data ******
test_acceleration = calculate_acceleration(test_velocity)

# combined_results = []
# for i,( v_track, a_track) in enumerate(zip(test_velocity, test_acceleration)):
#     per_sample_classes, overall_class = naive_bayes_classification_updated(v_track, a_track, likelihood, transition_Matrix, bird_mu, bird_std, airplane_mu, airplane_std)
#     print("Running velocity and acceleration feature")
#     print(f"Track{i+1}: Final_class = {overall_class}")
#     combined_results.append((per_sample_classes, overall_class))


# predicted_labels = [res[1] for res in combined_results]

# Calcualte Accuracy
# correct = sum(p == t for p, t in zip(predicted_labels, training_labels))
# accuracy = correct / len(test_labels)
# print(f"\nAccuracy: {accuracy * 100:.2f}%")

# print("Confusion Matrix")
# print(confusion_matrix(test_labels, predicted_labels, labels=['a', 'b']))

# print('\nClassification Report:')
# print(classification_report(training_labels, predicted_labels, target_names=['airplane', 'bird']))




# ****** Plotting Acceleration Likelihoods *******

# Clean NaNs (if not already cleaned)
def plotting_acceleration_likelihood(bird_acceleration, airplane_acceleration):
    bird_acc_clean = bird_acceleration[~np.isnan(bird_acceleration)]
    airplane_acc_clean = airplane_acceleration[~np.isnan(airplane_acceleration)]

    # Histogram and PDF overlay
    plt.figure(figsize=(10, 6))

    # Plot histograms
    plt.hist(bird_acc_clean, bins=100, density=True, alpha=0.6, label='Bird Acceleration', color='skyblue')
    plt.hist(airplane_acc_clean, bins=100, density=True, alpha=0.6, label='Airplane Acceleration', color='salmon')

    # Generate x values for PDF curves
    x = np.linspace(min(bird_acc_clean.min(), airplane_acc_clean.min()),
        max(bird_acc_clean.max(), airplane_acc_clean.max()), 1000)

    # Overlay normal distributions
    plt.plot(x, norm.pdf(x, bird_mu, bird_std), label=f'Bird PDF (μ={bird_mu:.2f}, σ={bird_std:.2f})', color='blue', linewidth=2)
    plt.plot(x, norm.pdf(x, airplane_mu, airplane_std), label=f'Airplane PDF (μ={airplane_mu:.2f}, σ={airplane_std:.2f})', color='red', linewidth=2)

    plt.title('Acceleration Likelihoods for Birds and Airplanes', fontsize = 25)
    plt.xlabel('Acceleration (Δvelocity)', fontsize = 25)
    plt.ylabel('Probability Density', fontsize = 25)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.legend(fontsize = 18)
    plt.grid(True)
    plt.tight_layout()
    # plt.show()

# ****** Plotting Displacement Likelihoods *******

def plotting_total_displacement(bird_displacement, airplane_displacement, bird_mu_d, bird_std_d, airplane_mu_d, airplane_std_d):
    
    bird_dis_clean = bird_displacement[~np.isnan(bird_displacement)]
    airplane_dis_clean = airplane_displacement[~np.isnan(airplane_displacement)]

    # Histogram and PDF overlay
    plt.figure(figsize=(10, 6))

    # Plot histograms
    plt.hist(bird_dis_clean, bins=100, density=True, alpha=0.6, label='Bird Displacement', color='skyblue')
    plt.hist(airplane_dis_clean, bins=100, density=True, alpha=0.6, label='Airplane Displacement', color='salmon')

    # Generate x values for PDF curves
    x = np.linspace(min(bird_dis_clean.min(), airplane_dis_clean.min()),
        max(bird_dis_clean.max(), airplane_dis_clean.max()), 1000)

    # Overlay normal distributions
    plt.plot(x, norm.pdf(x, bird_mu_d, bird_std_d), label=f'Bird PDF (μ={bird_mu_d:.2f}, σ={bird_std_d:.2f})', color='blue', linewidth=2)
    plt.plot(x, norm.pdf(x, airplane_mu_d, airplane_std_d), label=f'Airplane PDF (μ={airplane_mu_d:.2f}, σ={airplane_std_d:.2f})', color='red', linewidth=2)

    plt.title('Total Displacement Likelihoods for Birds and Airplanes', fontsize = 25)
    plt.xlabel('Total Displacement (m)', fontsize = 25)
    plt.ylabel('Probability Density', fontsize = 25)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    plt.legend(fontsize = 18)
    plt.grid(True)
    plt.tight_layout()
    # plt.show()



# *****Extra experiment ****

def compute_displacement(velocity):
    return np.nancumsum(np.nan_to_num(velocity, nan = 0))

bird_displacement = [compute_displacement(track) for track in velocity_data[:10]]
airplane_displacement = [compute_displacement(track) for track in velocity_data[10:]]


bird_displacement_all = np.concatenate(bird_displacement)
airplane_displacement_all = np.concatenate(airplane_displacement)

bird_mu_d, bird_std_d = norm.fit(bird_displacement)
airplane_mu_d, airplane_std_d = norm.fit(airplane_displacement)


# ****** Adding dispalcement in NBC ******

# bird_mu, bird_std, airplane_mu, airplane_std = calculate_likelihood(bird_displacement, airplane_displacement)
def naive_bayes_classification_updated_dis(velocity, acceleration, likelihood, transition_Matrix, bird_mu, bird_std, airplane_mu, airplane_std, bird_mu_d, bird_std_d, airplane_mu_d, airplane_std_d):

    prior_bird = 0.5
    prior_airplane = 0.5

    posterior_bird = prior_bird
    posterior_airplane = prior_airplane
    predictions = []
    displacement_track = np.nancumsum(np.nan_to_num(velocity, nan=0.0))

    for v, a in zip(velocity, acceleration):
        if np.isnan(v) and np.isnan(a):
            predictions.append('b' if posterior_bird > posterior_airplane else 'a')
            continue
        d = displacement_track[i]
        idx = velocity_to_index(v)
        # velocity likelihoods
        v_likelihood_bird = 1 if np.isnan(v) else likelihood[0, idx]
        v_likelihood_airplane = 1 if np.isnan(v) else likelihood[1, idx]

        # Acceleration likelihoods
        a_likelihood_bird = 1 if np.isnan(a) else norm.pdf(a, bird_mu, bird_std)
        a_likelihood_airplane = 1 if np.isnan(a) else norm.pdf(a, airplane_mu, airplane_std)

        #  Displacemenr likelihoods
        dist_likelihood_bird = 1 if np.isnan(a) else norm.pdf(d, bird_mu_d, bird_std_d)
        dist_likelihood_airplane = 1 if np.isnan(a) else norm.pdf(d, airplane_mu_d, airplane_std_d)

        # Combined likelihoods
        combined_bird_likelihood = v_likelihood_bird * a_likelihood_bird * dist_likelihood_bird
        combined_airplane_likelihood = v_likelihood_airplane * a_likelihood_airplane * dist_likelihood_airplane
        
        # Updating priors
        update_priors_bird = posterior_bird * transition_Matrix[0,0] + posterior_airplane*transition_Matrix[0,1]   # previous posterior becomes prior
        update_priors_airplane = posterior_airplane * transition_Matrix[1,1] + posterior_bird * transition_Matrix[1,0]

        # Updating posteriors
        posterior_bird = update_priors_bird * combined_bird_likelihood
        posterior_airplane = update_priors_airplane * combined_airplane_likelihood

        total = posterior_airplane + posterior_bird

        if total == 0:
            posterior_airplane = 0.5
            posterior_bird = 0.5
        else:
            posterior_bird /= total
            posterior_airplane /= total
        
        predictions.append('b' if posterior_bird > posterior_airplane else 'a')

    bird_count = predictions.count('b')
    airplane_count = predictions.count('a')
    final_classification = 'b' if bird_count > airplane_count else 'a'
    return predictions, final_classification


def plot_all_results(results):
    # Number of test tracks
    num_tracks = len(testing_data)
    # plt.figure()
    # Setup subplots: adjust rows/columns to your preference
    rows = 5
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(15, 12), sharex=True, sharey=True)

    # Flatten the axes array for easier indexing
    axes = axes.flatten()

    for i in range(num_tracks):
        velocity_track = testing_data[i]
        per_sample_labels, final_label = results[i]

    # Convert predicted labels: 'b' -> 0, 'a' -> 1
        label_numeric = [0 if lbl == 'b' else 1 for lbl in per_sample_labels]

        ax = axes[i]

    # Plot velocity track
        ax.plot(velocity_track, label='Velocity', color='blue', linewidth=1)

    # Plot predicted labels as step function
        ax.step(range(len(label_numeric)), label_numeric, where='mid',
            label='Prediction (0=Bird, 1=Airplane)', color='red', alpha=0.7)
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize='small')

    # Track-level title
        ax.set_title(f'Track {i+1}: {"Bird" if final_label == "b" else "Airplane"}')
        ax.set_xlabel('Time (s)', fontsize = 12)
        ax.set_ylabel('Velocity / Label', fontsize = 12)
        ax.grid(True)

    # Add a common legend (once, using the first axis' labels)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize='large')

    # plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space for the legend
    # plt.title("Test Tracks with Predicted Labels", fontsize=14, y=1.02)
    plt.subplots_adjust(hspace=0.5)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    # plt.show()

def plot_predicted_results(results):
        # Number of test tracks
    num_tracks = len(testing_data)
    # plt.figure()
    # Setup subplots: adjust rows/columns to your preference
    rows = 5
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(15, 12), sharex=True, sharey=True)

    # Flatten the axes array for easier indexing
    axes = axes.flatten()

    for i in range(num_tracks):
        # velocity_track = testing_data[i]
        per_sample_labels, final_label = results[i]

    # Convert predicted labels: 'b' -> 0, 'a' -> 1
        label_numeric = [0 if lbl == 'b' else 1 for lbl in per_sample_labels]

        ax = axes[i]

    # Plot velocity track
        # ax.plot(velocity_track, label='Velocity', color='blue', linewidth=1)

    # Plot predicted labels as step function
        ax.step(range(len(label_numeric)), label_numeric, where='mid',
            label='Prediction (0=Bird, 1=Airplane)', color='red', alpha=0.7)
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize='small')

    # Track-level title
        ax.set_title(f'Track {i+1}: {"Bird" if final_label == "b" else "Airplane"}')
        ax.set_xlabel('Time (s)', fontsize = 12)
        ax.set_ylabel('Velocity / Label', fontsize = 12)
        ax.grid(True)

    # Add a common legend (once, using the first axis' labels)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize='large')

    # plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space for the legend
    plt.suptitle("Test Tracks with Predicted Labels", fontsize=14, y=1.02)
    plt.subplots_adjust(hspace=0.5)
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    # plt.show()



# # # Number of test tracks
# # num_tracks = len(testing_data)

# # # Setup subplots: adjust rows/columns to your preference
# # rows = 5
# # cols = 2
# # fig, axes = plt.subplots(rows, cols, figsize=(15, 12), sharex=True, sharey=True)

# # # Flatten the axes array for easier indexing
# # axes = axes.flatten()

# # for i in range(num_tracks):
# #     velocity_track = testing_data[i]
# #     per_sample_labels, final_label = results[i]

# #     # Convert predicted labels: 'b' -> 0, 'a' -> 1
# #     label_numeric = [0 if lbl == 'b' else 1 for lbl in per_sample_labels]

# #     ax = axes[i]

# #     # Plot velocity track
# #     # ax.plot(velocity_track, label='Velocity', color='blue', linewidth=1)

# #     # Plot predicted labels as step function
# #     ax.step(range(len(label_numeric)), label_numeric, where='mid',
# #             label='Prediction (0=Bird, 1=Airplane)', color='red', alpha=0.7)
# #     # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize='small')

# #     # Track-level title
# #     ax.set_title(f'Track {i+1}: {"Bird" if final_label == "b" else "Airplane"}')
# #     ax.set_xlabel('Time (s)', fontsize = 12)
# #     ax.set_ylabel('Velocity / Label', fontsize = 12)
# #     ax.grid(True)

# # Add a common legend (once, using the first axis' labels)
# handles, labels = axes[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='lower center', ncol=2, fontsize='large')

# # plt.tight_layout(rect=[0, 0.05, 1, 1])  # Leave space for the legend
# plt.suptitle("Test Tracks with Predicted Labels", fontsize=14, y=1.02)
# plt.subplots_adjust(hspace=0.5)
# plt.xticks(fontsize = 16)
# plt.yticks(fontsize = 16)
# plt.show()



if model == 1:
#  Velocity feature
    results = []
    print("Running Velocity feature")
    for i,( v_track) in enumerate(test_velocity):
        per_sample_classes, overall_class = naive_bayes_classification(v_track, likelihood, transition_Matrix)
        print(f"Track{i+1}: Final_class = {overall_class}")
        results.append((per_sample_classes, overall_class))

    predicted_labels = [res[1] for res in results]
    print(classification_report(test_labels, predicted_labels, target_names=['airplane', 'bird']))
    plot_all_results(results)
    plot_predicted_results(results)
    plt.show()
#  Velocity + acceleration  feature
elif model == 2:
    results = []
    for i,( v_track, a_track) in enumerate(zip(test_velocity, test_acceleration)):
        per_sample_classes, overall_class = naive_bayes_classification_updated(v_track, a_track, likelihood, transition_Matrix, bird_mu, bird_std, airplane_mu, airplane_std)
        print(f"Track{i+1}: Final_class = {overall_class}")
        results.append((per_sample_classes, overall_class))

    predicted_labels = [res[1] for res in results]
    print(classification_report(test_labels, predicted_labels, target_names=['airplane', 'bird']))
    plot_all_results(results)
    plot_predicted_results(results)
    plotting_acceleration_likelihood(bird_acceleration, airplane_acceleration)
    plt.show()
# Velocity + acceleration + displacement
else:
    results = []
    print("Running velocity, acceleration and displacement feature")

    for i,(v_track, a_track) in enumerate(zip(test_velocity, test_acceleration)):
        per_sample_classes, overall_class = naive_bayes_classification_updated_dis(v_track, a_track,
                                                                                likelihood, transition_Matrix,
                                                                                bird_mu, bird_std, airplane_mu, 
                                                                                airplane_std, bird_mu_d, bird_std_d, 
                                                                                airplane_mu_d, airplane_std_d)
        print(f"Track{i+1}: Final_class = {overall_class}")
        results.append((per_sample_classes, overall_class))
        predicted_labels = [res[1] for res in results]
    print(classification_report(test_labels, predicted_labels, target_names=['airplane', 'bird']))
    plot_all_results(results)
    plot_predicted_results(results)
    plotting_total_displacement(bird_displacement_all, airplane_displacement_all, bird_mu_d, bird_std_d, airplane_mu_d, airplane_std_d)
    plt.show()
# Generating Likelihood for Acceleration (Birds vs Airplane) using training data
# def calculate_likelihood(acceleration_data, results):
#     bird_acceleration = []
#     airplane_acceleration = []

#     for i, track in enumerate(acceleration_data):
#         if results[i][1] == 'b':
#             bird_acceleration.extend(track)
#         else:
#             airplane_acceleration.extend(track)
# # Generate Gaussian distributions
#     bird_acceleration = np.array(bird_acceleration)
#     airplane_acceleration = np.array(airplane_acceleration)

#     bird_acceleration = bird_acceleration[(~np.isnan(bird_acceleration))]
#     airplane_acceleration = airplane_acceleration[(~np.isnan(airplane_acceleration))]

#     bird_mu, bird_std = norm.fit(bird_acceleration)
#     airplane_mu, airplane_std = norm.fit(airplane_acceleration)
#     return airplane_mu, airplane_std, airplane_acceleration, bird_mu, bird_std, bird_acceleration

# Likelihood functions for each class 
    # bird_acce_liklehood = norm.pdf(bird_acceleration, bird_mu, bird_std)
    # airplane_acce_liklehood = norm.pdf(airplane_acceleration, airplane_mu, airplane_std)
    # return bird_acce_liklehood, airplane_acce_liklehood