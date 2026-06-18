from scipy.stats import wilcoxon
import numpy as np
import pandas as pd
def compute_mean_std(csv_performance_file):
    df = pd.read_csv(csv_performance_file)
    mean_metrics = df.mean()
    std_metrics = df.std()
    return mean_metrics, std_metrics

def wilcoxson_test(base_metrics, new_metrics):
    '''Performs Wilcoxon signed-rank test between two sets of performance metrics.'''
    stat, p_value = wilcoxon(base_metrics, new_metrics)
    print(f"Wilcoxon test statistic: {stat}, p-value: {p_value}")
    #Save statistic and p-value to a file results/wilcoxon_test_results.txt
    with open("./result/wilcoxon_test_results.txt", "w") as f:       
        f.write(f"Wilcoxon test statistic: {stat}, p-value: {p_value}\n")
    return stat, p_value

def generate_10_fold_performance_summary(csv_save_path):
    '''Computes mean and std of 5 fold validated performance metrics across different seeds and saves to a summary CSV.'''
    seeds = [6, 0 , 3 , 7, 20, 35, 29, 27, 17, 42]
    csv_path = "model/ecoli_prot5_rand_seed{}/metrics_results.csv"
    
    rows = []
    for seed in seeds:
        csv_file = csv_path.format(seed)
        print(f"Processing metrics for seed {seed} from file: {csv_file}")

        mean_metrics, std_metrics = compute_mean_std(csv_file)
        rows.append({
            'Seed': seed,
            'RMSE': f"{mean_metrics['RMSE']:.4f}",
            'RMSE_STD': f"{std_metrics['RMSE']:.4f}",
            'Kendall': f"{mean_metrics['Kendall']:.4f}",
            'Kendall_STD': f"{std_metrics['Kendall']:.4f}",
            'R2': f"{mean_metrics['R2']:.4f} ",
            'R2_STD': f"{std_metrics['R2']:.4f}",
            'Pearson': f"{mean_metrics['Pearson']:.4f}",
            'Pearson_STD': f"{std_metrics['Pearson']:.4f}"
        })
    
    average_rmse = np.mean([float(row['RMSE']) for row in rows])
    average_kendall = np.mean([float(row['Kendall']) for row in rows])
    average_r2 = np.mean([float(row['R2']) for row in rows])
    average_pearson = np.mean([float(row['Pearson']) for row in rows])
    
    average_rmse_std = np.mean([float(row['RMSE_STD']) for row in rows])
    average_kendall_std = np.mean([float(row['Kendall_STD']) for row in rows])
    average_r2_std = np.mean([float(row['R2_STD']) for row in rows])
    average_pearson_std = np.mean([float(row['Pearson_STD']) for row in rows])  
    
    rows.append({
        'Seed': 'Average',
        'RMSE': f"{average_rmse:.4f}",
        'RMSE_STD': f"{average_rmse_std:.4f}",
        'Kendall': f"{average_kendall:.4f}",
        'Kendall_STD': f"{average_kendall_std:.4f}",
        'R2': f"{average_r2:.4f} ",
        'R2_STD': f"{average_r2_std:.4f}",
        'Pearson': f"{average_pearson:.4f}",
        'Pearson_STD': f"{average_pearson_std:.4f}"
    })

    df1 = pd.DataFrame(rows)

    df1.to_csv(csv_save_path, index=False)
    print(f"10-fold performance summary saved to: {csv_save_path}")

if __name__ == "__main__":
    # generate_10_fold_performance_summary("result/seed_performance_summary.csv")

    RMSE_BASE_METRICS = [0.529, 0.526, 0.528, 0.533, 0.530, 0.529, 0.526, 0.528, 0.533, 0.530] #MSCMamba RMSE values
    NEW_MODEL_METRICS = [0.4815, 0.483, 0.4803, 0.4826, 0.4836, 0.4846, 0.4782, 0.4775, 0.4796, 0.4803] # our new model RMSE values across 10 seeds

    stats, p_value = wilcoxson_test(RMSE_BASE_METRICS, NEW_MODEL_METRICS)