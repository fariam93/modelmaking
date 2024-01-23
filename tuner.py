from kerastuner.tuners import RandomSearch

# Define the hypermodel
hypermodel = Pix2PixHyperModel(input_shape, mask_shape, num_bracelet_types)

# Initialize the tuner
tuner = RandomSearch(
    hypermodel,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    directory='hyperparam_tuning',
    project_name='pix2pix_tuning'
)

# Perform hyperparameter search (use your train_dataset and val_dataset)
tuner.search(train_dataset, validation_data=val_dataset, epochs=20, callbacks=[early_stopping_callback])

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]
