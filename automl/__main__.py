from DataReading.DataReader import ReadData
from DataScaling.DataScaler import ScaleData
from DataModeling.DataModeler import ModelData
import pandas as pd

import argparse
import warnings

warnings.filterwarnings('ignore')


p = argparse.ArgumentParser()
p.add_argument('--data_path', type=str, help='Mandatory. Path to the data file.')
p.add_argument('--target_var', type=str, help='Mandatory. Name of the target column.')
p.add_argument('--test_size', type=float, help='Optional. Size of the test set. Default is 0.2.')
# p.add_argument('--save_model', type=str, help='Path to save the model.')
# p.add_argument('--save_predictions', type=str, help='Path to save the predictions.')
p.add_argument('--path_to_save_metrics', type=str, help='Optional. Path to save the metrics. It must be a file with the extension .csv.')
args = p.parse_args()


data_reader = ReadData(args.data_path)
data_reader.read_data()


data_scaler = ScaleData(data_reader.data, target=args.target_var)
data_scaler.select_dtypes()
data_scaler.clean_data()
scaling_methods = data_scaler.create_scaling_methods_pool()


data_modeler = ModelData(data_scaler.X, data_scaler.y, test_size=args.test_size if args.test_size else 0.2)
models = data_modeler.create_models_pool()


results = {}

# Iterate over all the scaling methods and models
for scaling_method in scaling_methods:
    for model_name, model in models.items():
        print(f'Using {scaling_method} scaling method and {model_name} model.')

        # Scale the data
        xtr_scaled, xte_scaled = data_scaler.scale_data(scaler=scaling_methods[scaling_method], xtr=data_modeler.xtr, xte=data_modeler.xte)

        # Train the model and get the predictions
        predictions = data_modeler.train_algorithm_and_return_predictions(model, xtr_scaled=xtr_scaled, xte_scaled=xte_scaled)

        # Calculate the metrics
        metrics = data_modeler.evaluate_model(model=model, predictions=predictions)

        # Store the results
        results[(scaling_method, model_name)] = metrics


# Convert the results to a DataFrame
results_df = pd.DataFrame(results).T
if args.path_to_save_metrics:
    results_df.to_csv(args.path_to_save_metrics)



