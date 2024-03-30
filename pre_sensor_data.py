import re
from datetime import datetime
from collections import Counter
import numpy as np

def get(set="dataset0", seconds=3, choosing="nodding", sensor="accel"):
    path = f"C:\\Users\\xueyu\\Desktop\\evasion\\{set}\\{choosing}\\{sensor}.txt"
    frequency_x = int(count_entries_per_second(path))
    pattern = re.compile('')
    if sensor == "accel":
        pattern = re.compile(r'Accelerometer\[\d\] :([-\d.]+)')
    elif sensor == "gyro":
        pattern = re.compile(r'Gyroscope\[\d\] :([-\d.]+)')
    elif sensor == "mag":
        pattern = re.compile(r'Magnetometer\[\d\] :([-\d.]+)')
    data = []
    with open(path, "r") as file:
        for line in file:
            matches = pattern.findall(line)
            if matches:
                data.append([float(x) for x in matches])

    # trim the result to fit the model
    data = data[:-(len(data) % (seconds * frequency_x))]

    # To obtain a pool of samples,
    # we then ran a sliding window of fixed length (1, 3, 5, 10
    # and 15 seconds) with step size equal to 1/10th of its length
    # over the data of every recorded session and labeled it with the corresponding user.
    new_data = sliding_window(data, seconds, frequency_x)

    # # reshape the data
    # data = np.array(data).reshape(-1, seconds * frequency_x, 3)

    # # normalize the data
    # min_value, max_value = np.min(data), np.max(data)
    # data = (data - min_value) / (max_value - min_value)

    return new_data, frequency_x

def sliding_window(dataset, seconds, freq):
    dataset = np.array(dataset)
    # reshape the dataset
    dataset = dataset.reshape(-1, 3)
    print("dataset: ", dataset.shape)
    window = seconds * freq
    step = window // 10
    sliding_window = []
    for window_start in range(0, len(dataset) - window, step):
        window_end = window_start + window
        sliding_window.append(dataset[window_start:window_end])

    return np.array(sliding_window)


# Function to convert log timestamp to a simplified datetime object (ignoring milliseconds)
def parse_timestamp_to_second(hour, minute, second):
    # Use a fixed date for all entries since only time is relevant here
    return datetime(2000, 1, 1, int(hour), int(minute), int(second))

# Function to count log entries per second
def count_entries_per_second(file_path):
    with open(file_path, 'r') as file:
        content = file.readlines()

    timestamps = []
    # Regex to match the timestamp format
    # regex = r"Time: (\d{2}) (\d{2}) (\d{2}) \d{3}"
    regex = r"Log Entry :  Time: (\d{2}) (\d{2}) (\d{2}) \d{2,3}"
    
    for line in content:
        match = re.search(regex, line)
        if match:
            # Ignore milliseconds for this part
            timestamps.append(parse_timestamp_to_second(*match.groups()[:3]))

    # Count occurrences of each second
    counts = Counter(timestamps)
    
    # Calculate the total number of log entries and the number of unique seconds
    total_entries = sum(counts.values())
    unique_seconds = len(counts)
    
    # Calculate the average number of entries per second
    if unique_seconds > 0:
        average_entries_per_second = total_entries / unique_seconds
        print(f"Average count per second: {average_entries_per_second}")
    else:
        print("No log entries found.")
        average_entries_per_second = 0

    return average_entries_per_second
    
# count_entries_per_second("C:\\Users\\xueyu\\Desktop\\evasion\\dataset0\\shaking\\accel.txt")
if __name__ == "__main__":
    result = get(set="3rd_party_dataset", seconds=3, choosing="nodding", sensor="accel")
    print(result)
