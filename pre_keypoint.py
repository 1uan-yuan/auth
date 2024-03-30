import os
import json
import numpy as np

def extract_keypoints(directory):
    keypoints_list = []

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            
            try:
                with open(filepath, 'r') as file:
                    data = json.load(file)
                    people = data.get("people", [])
                    
                    if people:
                        keypoints = people[0].get("pose_keypoints_2d", [])
                        if keypoints:
                            first_two = keypoints[:2]
                            keypoints_list.append(first_two)
            
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    print(f"Processed {len(keypoints_list)} files")

    return keypoints_list

def get(set="dataset0", seconds=3, frequency_y=30, choosing="nodding"):
    directory= f'C:\\Users\\xueyu\\Desktop\\evasion\\{set}\\{choosing}_json'
    result = extract_keypoints(directory)

    # trim the result to fit the model
    result = result[:-(len(result) % (seconds * frequency_y))]
    result = np.array(result).reshape(-1, seconds * frequency_y, 2)

    # normalize the data
    min_value, max_value = np.min(result), np.max(result)
    result = (result - min_value) / (max_value - min_value)

    return result

def write_into_txt(target, name):
    with open(f"C:\\Users\\xueyu\\Desktop\\evasion\\dataset\\{name}.txt", "w") as file:
        for pairs in target:
            for pair in pairs:
                file.write(f"{pair}\n")

if __name__ == "__main__":
    nodding = get(choosing="nodding")
    shaking = get(choosing="shaking")