# experiment_runner.py
import yaml
import pandas as pd
import os
import logging
import time
from datetime import datetime
import torch
from torch.utils.data import DataLoader

import nets
from mydatasets import adult
from selection.random_selection import RandomSelection
from selection.kmeans import KMeansSelection  
from selection.cg_influence import CGInfluenceSelection
from selection.tracin import TracInSelection
from selection.arnoldi import ArnoldiSelection
from selection.dvrl import DVRLSelection
# Can continue importing other methods...

from model_utils import train_model, evaluate_model

def load_config(config_path="experiments.yaml"):
    """Load experiment configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_selector(method_name):
    """Get the corresponding selector class based on method name"""
    method_map = {
        'random': RandomSelection,
        'kmeans': KMeansSelection,
        'cg_influence': CGInfluenceSelection,
        # Can continue adding other methods
        # 'glister': GlisterSelection,
        # 'gradmatch': GradMatchSelection,
        # ...
        'tracin': TracInSelection,
        'arnoldi': ArnoldiSelection,
        'dvrl': DVRLSelection,
    }
    if method_name not in method_map:
        raise ValueError(f"Unknown method: {method_name}. Available methods: {list(method_map.keys())}")
    return method_map[method_name]()

def setup_directories(config):
    """Create necessary directories"""
    dirs = [
        config['results']['save_path'],
        config['results']['log_path'], 
        config['results']['models_path']
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def setup_experiment_logging(exp_name, log_path):
    """Set up logging for a single experiment"""
    # log_file is now the full path passed from run_single_experiment
    log_file = log_path
    
    # Create new logger
    logger = logging.getLogger(exp_name)
    logger.setLevel(logging.INFO)
    
    # Clear previous handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def save_results_to_csv(method, selection_fraction, precision, recall, f1, accuracy, 
                       selection_time, training_time, total_time, csv_path):
    """
    Save experiment results to CSV file, supporting append mode and MultiIndex format
    
    Args:
        method: Method name
        selection_fraction: Selection fraction
        precision, recall, f1, accuracy: Performance metrics
        selection_time: Data selection time
        training_time: Training time
        total_time: Total time
        csv_path: CSV file path
    """
    # Ensure directory exists
    if not os.path.exists(os.path.dirname(csv_path)):
        os.makedirs(os.path.dirname(csv_path))

    selection_ratio = str(selection_fraction)
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'selection_time', 'training_time', 'total_time']
    columns = pd.MultiIndex.from_product([[selection_ratio], metrics])

    # Convert to string for appending, keeping specified decimal places
    accuracy_str = f"{accuracy:.4f}"
    precision_str = f"{precision:.4f}"
    recall_str = f"{recall:.4f}"
    f1_str = f"{f1:.4f}"
    selection_time_str = f"{selection_time:.2f}"
    training_time_str = f"{training_time:.2f}"
    total_time_str = f"{total_time:.2f}"

    new_data = pd.DataFrame([[accuracy_str, precision_str, recall_str, f1_str, 
                             selection_time_str, training_time_str, total_time_str]], 
                           index=[method], columns=columns)

    if os.path.exists(csv_path):
        # Read existing data
        old_df = pd.read_csv(csv_path, header=[0, 1], index_col=0)
        updated_df = old_df.copy()

        # Add new columns (if not exist)
        for col in new_data.columns:
            if col not in updated_df.columns:
                updated_df[col] = pd.NA

        # Add new row (if not exist)
        if method not in updated_df.index:
            updated_df.loc[method] = pd.NA

        # Update data: append | separated values
        for col in new_data.columns:
            old_val = str(updated_df.loc[method, col]) if pd.notna(updated_df.loc[method, col]) else ""
            new_val = new_data.loc[method, col]
            if old_val and old_val != 'nan':
                # Check if same value already exists, avoid duplicates
                if new_val not in old_val.split('|'):
                    updated_df.loc[method, col] = old_val + '|' + new_val
            else:
                updated_df.loc[method, col] = new_val

        updated_df.to_csv(csv_path)
    else:
        # Create new file
        new_data.to_csv(csv_path)

def run_single_experiment(exp_config, general_config, datasets):
    """Run a single experiment"""
    exp_name = exp_config['name']
    method = exp_config['method']
    
    print(f"\n{'='*60}")
    print(f"Starting experiment: {exp_name} (Method: {method})")
    print(f"{'='*60}")
    
    # Construct the specific log file path for this experiment
    dataset_name = general_config['general']['dataset']
    model_name = general_config['general']['model_name']
    log_dir = os.path.join(general_config['results']['save_path'], dataset_name, method)
    # Use exp_name as it already includes model and fraction in its naming convention
    specific_log_file = os.path.join(log_dir, f"{exp_name}.log")
    
    # Ensure the specific log directory exists
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup logging with the specific file path
    logger = setup_experiment_logging(exp_name, specific_log_file)

    logger.info(f"Log file: {specific_log_file}")

    start_time = time.time()
    
    try:
        # Set device
        if general_config['general']['device'] == 'auto':
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(general_config['general']['device'])
        
        logger.info(f"Using device: {device}")
        logger.info(f"Experiment config: {exp_config}")
        
        # Create model
        dst_train, dst_val, dst_test = datasets
        
        # Correctly get data dimensions
        sample_data = dst_train[0][0]
        if len(sample_data.shape) == 1:  # Tabular data
            input_dim = sample_data.shape[0]
            channel = 1
            im_size = input_dim
        else:  # Image data
            channel, im_size = sample_data.shape[0], sample_data.shape[-1]
            input_dim = channel * im_size * im_size if len(sample_data.shape) == 3 else channel * im_size
        
        # Get number of classes
        num_classes = len(set([dst_train[i][1] for i in range(min(1000, len(dst_train)))]))
        
        logger.info(f"Data dimensions: input_dim={input_dim}, num_classes={num_classes}")
        
        base_model = nets.__dict__[general_config['general']['model_name']](
            channel=channel,
            num_classes=num_classes,
            im_size=im_size,
            pretrained=False
        )
        
        # Get selector and run selection
        selector = get_selector(method)
        
        # Prepare method-specific parameters
        method_params = {k: v for k, v in exp_config.items() 
                        if k not in ['name', 'method', 'selection_fraction']}
        
        logger.info(f"Starting data selection...")
        selection_start_time = time.time()
        
        core_loader = selector.select(
            model=base_model,
            selection_fraction=exp_config['selection_fraction'],
            train_set=dst_train,
            task_type=general_config['general']['task_type'],
            device=device,
            **method_params
        )
        
        selection_end_time = time.time()
        selection_time = selection_end_time - selection_start_time
        
        logger.info(f"Data selection completed, core set size: {len(core_loader.dataset)}, selection time: {selection_time:.2f}s")
        
        # Train model
        best_model_path = os.path.join(
            general_config['results']['models_path'], 
            f"{exp_name}.pth"
        )
        
        logger.info(f"Starting model training...")
        training_start_time = time.time()
        
        trained_model = train_model(
            model=base_model,
            train_loader=core_loader,
            val_loader=DataLoader(dst_val, batch_size=general_config['general']['batch_size'], shuffle=False),
            device=device,
            task_type=general_config['general']['task_type'],
            epochs=general_config['general']['epochs'],
            patience=general_config['general']['patience'],
            lr=general_config['general']['selection_lr'],
            momentum=general_config['general']['selection_momentum'],
            weight_decay=general_config['general']['selection_weight_decay'],
            best_model_path=best_model_path
        )
        
        training_end_time = time.time()
        training_time = training_end_time - training_start_time
        
        # Evaluate model
        logger.info(f"Starting model evaluation...")
        test_loader = DataLoader(dst_test, batch_size=general_config['general']['batch_size'], shuffle=False)
        
        total_time = time.time() - start_time
        
        if general_config['general']['task_type'] == 'classification':
            acc, precision, recall, f1 = evaluate_model(
                model=trained_model,
                best_model_path=best_model_path,
                test_loader=test_loader,
                device=device,
                task_type=general_config['general']['task_type']
            )
            
            # Save to CSV file (new format)
            csv_path = os.path.join(
                general_config['results']['save_path'], 
                general_config['general']['dataset'],
                general_config['general']['model_name'] + '_new.csv'
            )
            
            save_results_to_csv(
                method=method,
                selection_fraction=exp_config['selection_fraction'],
                precision=precision,
                recall=recall,
                f1=f1,
                accuracy=acc,
                selection_time=selection_time,
                training_time=training_time,
                total_time=total_time,
                csv_path=csv_path
            )
            
            results = {
                'experiment_name': exp_name,
                'method': method,
                'selection_fraction': exp_config['selection_fraction'],
                'core_set_size': len(core_loader.dataset),
                'accuracy': acc,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'selection_time': selection_time,
                'training_time': training_time,
                'total_time': total_time,
                'status': 'success'
            }
        else:
            mse, mae = evaluate_model(
                model=trained_model,
                best_model_path=best_model_path,
                test_loader=test_loader,
                device=device,
                task_type=general_config['general']['task_type']
            )
            
            results = {
                'experiment_name': exp_name,
                'method': method,
                'selection_fraction': exp_config['selection_fraction'],
                'core_set_size': len(core_loader.dataset),
                'mse': mse,
                'mae': mae,
                'selection_time': selection_time,
                'training_time': training_time,
                'total_time': total_time,
                'status': 'success'
            }
        
        logger.info(f"Experiment {exp_name} completed successfully! Total time: {total_time:.2f}s (Selection: {selection_time:.2f}s, Training: {training_time:.2f}s)")
        # Log experiment results
        logger.info(f"Experiment results: {results}")
        return results
        
    except Exception as e:
        logger.error(f"Experiment {exp_name} failed: {str(e)}")
        total_time = time.time() - start_time
        return {
            'experiment_name': exp_name,
            'method': method,
            'selection_fraction': exp_config.get('selection_fraction', 0),
            'status': 'failed',
            'error': str(e),
            'total_time': total_time
        }

def run_experiments(config_path="experiments.yaml"):
    """Run batch experiments"""
    print("Loading configuration file...")
    config = load_config(config_path)
    
    # Setup directories
    setup_directories(config)
    
    # Load dataset
    print("Loading dataset...")
    if config['general']['dataset'] == 'adult':
        channel, im_size, num_classes, class_names, mean, std, dst_train, dst_val, dst_test = adult(
            config['general']['data_path'], 
            config['general']['balance_target']
        )
        datasets = (dst_train, dst_val, dst_test)
    else:
        raise ValueError(f"Unsupported dataset: {config['general']['dataset']}")
    
    # Run all experiments
    all_results = []
    successful_experiments = 0
    
    for exp_config in config['experiments']:
        result = run_single_experiment(exp_config, config, datasets)
        all_results.append(result)
        
        if result.get('status') == 'success':
            successful_experiments += 1
    
    # Save results (original format, for compatibility)
    results_df = pd.DataFrame(all_results)
    results_file = os.path.join(
        config['results']['save_path'], 
        config['results']['comparison_file']
    )
    results_df.to_csv(results_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"All experiments completed!")
    print(f"Successful: {successful_experiments}/{len(config['experiments'])}")
    print(f"Results saved to: {results_file}")
    
    # New format CSV file path
    new_csv_path = os.path.join(
        config['results']['save_path'], 
        config['general']['dataset'],
        config['general']['model_name'] + '_new.csv'
    )
    print(f"Detailed results (new format) saved to: {new_csv_path}")
    print(f"{'='*60}")
    
    # Display results summary
    if successful_experiments > 0:
        successful_results = results_df[results_df['status'] == 'success']
        if config['general']['task_type'] == 'classification':
            print("\nExperiment Results Summary (sorted by F1-score):")
            summary_columns = ['experiment_name', 'method', 'selection_fraction', 'accuracy', 'f1_score', 'selection_time', 'training_time']
            summary = successful_results[summary_columns].sort_values('f1_score', ascending=False)
        else:
            print("\nExperiment Results Summary (sorted by MSE):")
            summary_columns = ['experiment_name', 'method', 'selection_fraction', 'mse', 'mae', 'selection_time', 'training_time']
            summary = successful_results[summary_columns].sort_values('mse')
        
        print(summary.round(4).to_string(index=False))
    
    return results_df

if __name__ == "__main__":
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else "experiments.yaml"
    run_experiments(config_file) 