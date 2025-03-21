import numpy as np
import os 
import time
from pathlib import Path
import sys


def load_data(filename):
    filepath = os.path.join("datasets", filename)
    data = []
    with open(filepath, 'r') as f:  
        for line in f:
            parts = list(filter(None, line.strip().split()))
            converted = [float(x) for x in parts]
            data.append(converted)
    return np.array(data)

def leave_one_out_accuracy(data):
   #code as from lecture viedo
    n = data.shape[0]
    correct = 0
    
    for i in range(n):
        test_label = data[i, 0]
        test_features = data[i, 1:]
        
        min_dist = float('inf')
        nearest_label = None
        
        for j in range(n):
            if i == j:  #leave current feature out
                continue
            
            train_label = data[j, 0]
            train_features = data[j, 1:]
            
            # euclidian distance(keep dquared for accueacy)
            dist_sq = np.sum((test_features - train_features)**2)
            
            # update nearest neighbor(also break tie)
            if dist_sq < min_dist or (dist_sq == min_dist and j < nearest_label):
                min_dist = dist_sq
                nearest_label = train_label
        
        if nearest_label == test_label:
            correct += 1
    
    return (correct / n) * 100

def vectorized_looc_accuracy(data):
    n = data.shape[0]
    features = data[:, 1:]
    labels = data[:, 0]
    correct = 0
    
    for i in range(n):
        # leave current one out
        mask = np.arange(n) != i
        
        # calculate all distance(squared)
        diff = features[i] - features[mask]
        distances_sq = np.sum(diff**2, axis=1)
        
        # find smallest distance
        nearest_idx = np.argmin(distances_sq)
        
        if labels[mask][nearest_idx] == labels[i]:
            correct += 1
    
    return (correct / n) * 100

def forward_selection(data):
    n_samples, n_features = data.shape[0], data.shape[1]-1  # first column as class
    best_features = []
    best_acc = 0.0
    history = []  # tracing
    
    print("\nStarting Forward Selection...")
    
    # trace through all features
    for step in range(1, n_features+1):
        current_acc = 0.0
        candidate_feature = None
        candidate_acc = 0.0
        
        # trace all un-visted features
        for feature in range(1, n_features+1):  
            if feature in best_features:
                continue
                
            current_set = [0] + best_features + [feature]
            
            # calculate accuracy
            acc = vectorized_looc_accuracy(data[:, current_set])
            
            print(f"   Testing feature {feature} ➔ Accuracy: {acc:.1f}%")
            
            # update best current feature
            if acc > candidate_acc:
                candidate_acc = acc
                candidate_feature = feature
                
        # beast feature of current round
        if candidate_acc > best_acc:
            best_acc = candidate_acc
            best_features.append(candidate_feature)
            history.append({
                'step': step,
                'added_feature': candidate_feature,
                'accuracy': candidate_acc,
                'current_features': best_features.copy()
            })
            print(f"\n Step {step}: Added feature {candidate_feature}")
            print(f"   Current features: {best_features} ➔ Accuracy: {best_acc:.1f}%")
        else:
            print(f"\n Step {step}: No improvement. Final accuracy: {best_acc:.1f}%")
            break  # stop when there's no better accuracy
    
    return best_features, best_acc, history

import numpy as np

def forward_selection_full(data): #same but doesn't stop early
    n_samples, n_features = data.shape[0], data.shape[1]-1
    selected = []
    best_acc = 0.0
    history = []
    
    print(f"\nStarting Forward Selection (Full Search)")
    print(f"Total features to evaluate: {n_features}")
    
    for step in range(1, n_features+1):
        print(f"\n=== Level {step} ===")
        candidates = []
        
        for feature in range(1, n_features+1):
            if feature in selected:
                continue
                
            candidate_set = [0] + selected + [feature]
            
            acc = vectorized_looc_accuracy(data[:, candidate_set])
            
            candidates.append((feature, acc))
            print(f"  Testing feature {feature} ➔ Accuracy: {acc:.2f}%")
        
        if candidates:
            best_feature = max(candidates, key=lambda x: x[1])
            selected.append(best_feature[0])
            current_acc = best_feature[1]
            
            if current_acc > best_acc:
                best_acc = current_acc
            else:
                print(f"  Accuracy dropped from {best_acc:.2f}% to {current_acc:.2f}%")
            
            history.append({
                'step': step,
                'feature': best_feature[0],
                'accuracy': current_acc,
                'best_global_acc': best_acc,
                'selected': selected.copy()
            })
    
    return selected, best_acc, history

def backward_elimination(data):
    n_samples, n_features = data.shape[0], data.shape[1]-1  # 1st column as class
    remaining_features = list(range(1, n_features+1))       # init with all features
    best_acc = vectorized_looc_accuracy(data)               # init with accuracy
    history = [{
        'step': 0,
        'remaining_features': remaining_features.copy(),
        'accuracy': best_acc,
        'removed_feature': None
    }]

    print(f"\nStarting Backward Elimination (Full Search)")
    print(f"Initial accuracy with all features: {best_acc:.1f}%")

    # loop all possiable eliminations
    for step in range(1, n_features):
        current_worst_feature = None
        current_best_acc = 0.0

        # loop rest of features, try removing all of them
        for feature in remaining_features:
            # potential features（except current one）
            candidate_features = [f for f in remaining_features if f != feature]
            candidate_set = [0] + candidate_features

            # calculate accuracy
            acc = vectorized_looc_accuracy(data[:, candidate_set])
            print(f"  Testing remove feature {feature} ➔ Accuracy: {acc:.1f}%")

            # keep track of best accuracy
            if acc > current_best_acc:
                current_best_acc = acc
                current_worst_feature = feature

        # which feature to remove now
        if current_worst_feature is not None:
            remaining_features.remove(current_worst_feature)
            best_acc = max(best_acc, current_best_acc)
            history.append({
                'step': step,
                'remaining_features': remaining_features.copy(),
                'accuracy': current_best_acc,
                'removed_feature': current_worst_feature
            })
            print(f"\n Step {step}: Removed feature {current_worst_feature}")
            print(f"   Current features: {remaining_features} ➔ Accuracy: {current_best_acc:.1f}%")
        else:
            break  

    return remaining_features, best_acc, history

# def main():
#     filename = input("Enter dataset name (e.g. Large_Data__5.txt): ")
#     data = load_data(filename)

    # with open("output.txt", "w") as f:
    #     sys.stdout = f  # Redirect output to file
    
    #     print(f"\nLoaded {data.shape[0]} instances with {data.shape[1]-1} features")

    #     # lecture version of leave-one-out accuracy
    #     print("\nValidating with strict MATLAB equivalence...")
    #     baseline_acc = leave_one_out_accuracy(data)
    #     print(f"MATLAB-style Accuracy: {baseline_acc:.2f}%")
        
    #     # my own version of accuracy
    #     print("\nRunning optimized version...")
    #     vectorized_acc = vectorized_looc_accuracy(data)
    #     print(f"Vectorized Accuracy: {vectorized_acc:.2f}%")
        
    #     # do forward selection
    #     selected_features, final_acc, history = forward_selection(data)
        
    #     print("\n\n=== Final Result(forward) ===")
    #     print(f"Optimal feature subset: {selected_features}")
    #     print(f"Maximum accuracy achieved: {final_acc:.1f}%")
        
    #     # trace history
    #     print("\n=== Search History(forward) ===")
    #     for record in history:
    #         print(f"Step {record['step']:2d} | +Feature {record['added_feature']:2d} "
    #             f"| Acc: {record['accuracy']:.1f}% | Current: {record['current_features']}")
        
    #     # do forward selection(full)
    #     selected_features, final_acc, history = forward_selection_full(data)
        
    #     print("\n\n=== Final Result(forward full) ===")
    #     print(f"Optimal feature subset: {selected_features}")
    #     print(f"Maximum accuracy achieved: {final_acc:.1f}%")

    #     # do backward elimination
    #     remaining_features, final_acc, history = backward_elimination(data)
        
    #     print("\n\n=== Final Result(backward) ===")
    #     print(f"Optimal feature subset: {remaining_features}")
    #     print(f"Maximum accuracy achieved: {final_acc:.1f}%")

    #     # trace history
    #     print("\n=== Elimination History ===")
    #     for record in history:
    #         if record['step'] == 0:
    #             print(f"Initial | All features | Acc: {record['accuracy']:.1f}%")
    #         else:
    #             print(f"Step {record['step']:2d} | -Feature {record['removed_feature']:2d} "
    #                 f"| Acc: {record['accuracy']:.1f}% | Remaining: {record['remaining_features']}")
    
    #     sys.stdout = sys.__stdout__  # Reset output back to console
    #     print("Execution completed. Results saved in output.txt.")

def main():
    filename = input("Type in the name of the file to test: ")
    data = load_data(filename)
    
    with open("output.txt", "w") as f:
        sys.stdout = f  # Redirect output to file
        
        print("\nWelcome to Ann Xie's Feature Selection Algorithm.")
        print("\nThis dataset has {} features (not including the class attribute), with {} instances.".format(data.shape[1]-1, data.shape[0]))
        
        print("\nRunning nearest neighbor with all features, using 'leave-one-out' evaluation, I get an accuracy of {:.1f}%".format(vectorized_looc_accuracy(data)))
        
        selected_features, final_acc, history = forward_selection_full(data)
        selected_features, final_acc, history = backward_elimination(data)
        
        sys.stdout = sys.__stdout__  # Reset output back to console
        print("Execution completed. Results saved in output.txt.")

if __name__ == "__main__":
    main()