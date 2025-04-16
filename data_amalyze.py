# Import necessary libraries
import os
import matplotlib.pyplot as plt


def process_data(file_path):
    """
    Process the data from the given txt file and calculate accuracy for each true value.

    Args:
        file_path (str): Path to the txt file containing the data.

    Returns:
        dict: A dictionary with true values as keys and their respective accuracies as values.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, 'r') as file:
        data = file.read().strip().split('\n')  # Split data into segments by '\n'

    true_counts = {}
    correct_counts = {}

    for segment in data:
        # Split each segment by ',' and convert to integers
        predict, true = map(int, segment.split(','))
        if true not in true_counts:
            true_counts[true] = 0
            correct_counts[true] = 0
        true_counts[true] += 1
        if predict == true:
            correct_counts[true] += 1

    # Calculate accuracy for each true value
    accuracy = {true: correct_counts[true] /
                true_counts[true] for true in true_counts}

    return accuracy


def compare_accuracies(accuracy_1, accuracy_2):
    """
    Compare the accuracies of two datasets for each true value.

    Args:
        accuracy_1 (dict): Accuracy dictionary for the first dataset.
        accuracy_2 (dict): Accuracy dictionary for the second dataset.

    Returns:
        dict: A dictionary with true values as keys and the difference in accuracy as values.
    """
    comparison = {}
    for true_value in accuracy_1:
        acc1 = accuracy_1.get(true_value, 0)
        acc2 = accuracy_2.get(true_value, 0)
        comparison[true_value] = (acc2 - acc1) * \
            100  # Difference in percentage
    return comparison


def plot_comparison(comparison):
    """
    Plot a bar chart for the accuracy differences.

    Args:
        comparison (dict): A dictionary with true values as keys and the difference in accuracy as values.
    """
    # Sort the comparison dictionary by accuracy difference in descending order
    sorted_comparison = sorted(
        comparison.items(), key=lambda x: x[1], reverse=True)

    # Extract values and differences
    values = [item[0] for item in sorted_comparison]
    differences = [item[1] for item in sorted_comparison]

    # Plot the bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(values)), differences,
            color='skyblue', edgecolor='black')
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Accuracy Difference (%)', fontsize=12)
    plt.title('Accuracy Difference by Value', fontsize=14)
    plt.xticks(range(len(values)), values, fontsize=10,
               rotation=45)  # Display all values on the x-axis
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot as a high-quality image
    plt.savefig('accuracy_difference_plot.png', dpi=300)
    plt.show()


# Example usage
if __name__ == "__main__":
    file_path_1 = "compared_right.txt"  # Replace with the actual file path
    file_path_2 = "ntu_cv_aagcn_joint_right.txt"  # Replace with the actual file path

    try:
        accuracy_1 = process_data(file_path_1)
        accuracy_2 = process_data(file_path_2)
        comparison = compare_accuracies(accuracy_1, accuracy_2)

        # Sort and display the comparison
        sorted_comparison = sorted(
            comparison.items(), key=lambda x: x[1], reverse=True)
        for true_value, diff in sorted_comparison:
            print(
                f"True Value: {true_value}, Accuracy Difference: {diff:+.1f}%")

        # Plot the comparison
        plot_comparison(comparison)
    except Exception as e:
        print(f"Error: {e}")
