from DataReading.DataReader import ReadData
from DataScaling.DataScaler import ScaleData
from DataModeling.DataModeler import ModelData


data_reader = ReadData('./test_data/Maths.csv')


data_scaler = ScaleData(data_reader.data, data_reader.target, scaling_method='standard', scaling_params=None)
numerical_data = data_scaler.select_dtypes()
cleaned_numerical_data = data_scaler.clean_data()
scaling_methods = data_scaler.create_scaling_methods_pool()


model_data = ModelData(data_scaler.train_data, data_scaler.target_data, test_size=0.2)
models = model_data.create_models_pool()


results = {}

# Iterate over all the scaling methods and models
for scaling_method in scaling_methods:
    for model_name, model in models.items():
        print(f'Using {scaling_method} scaling method and {model_name} model.')

        # Scale the data
        xtr_scaled, ytr_scaled = data_scaler.scale_data(scaler=scaling_methods[scaling_method])

        # Train the model and get the predictions
        predictions = model_data.train_algorithm_and_return_predictions(model)

        # Calculate the metrics
        metrics = data_scaler.calculate_metrics(predictions)

        # Store the results
        results[(scaling_method, model_name)] = metrics


# Print the results
for key, value in results.items():
    print(f'{key[0]} scaling method and {key[1]} model: {value}')
    


