# Hyperparameters

## REDD

| Models                 | Sequence Length | Batch Size | Epoch | Learning Rate | Appliance |
|------------------------|-----------------|------------|-------|---------------|-----------|
| S2P + Homoskedastic    | 99              | 32         | 50    | 1e-3          | Fridge    |
| S2P + Heteroskedastic  | 99              | 1024       | 50    | 1e-4          | Fridge    |
| S2P + Homoskedastic    | 99              | 32         | 50    | 1e-3          | Dishwasher|
| S2P + Heteroskedastic  | 99              | 1024       | 50    | 1e-4          | Dishwasher|
| S2P + Homoskedastic    | 99              | 32         | 50    | 1e-3          | Microwave |
| S2P + Heteroskedastic  | 99              | 1024       | 50    | 1e-4          | Microwave |

## Pecan Street

| Models                 | Sequence Length | Batch Size | Epoch | Learning Rate | Appliance |
|------------------------|-----------------|------------|-------|---------------|-----------|
| S2P + Homoskedastic    | 99              | 1024       | 50    | 1e-4          | Fridge    |
| S2P + Heteroskedastic  | 99              | 1024       | 50    | 1e-4          | Fridge    |
| S2P + Homoskedastic    | 99              | 2048       | 50    | 1e-4          | AC        |
| S2P + Heteroskedastic  | 99              | 2048       | 50    | 1e-4          | AC        |
| S2P + Homoskedastic    | 99              | 2048       | 50    | 1e-5          | Furnace   |
| S2P + Heteroskedastic  | 99              | 2048       | 50    | 1e-5          | Furnace   |

## REFIT 

| Models                 | Sequence Length | Batch Size | Epoch | Learning Rate | Appliance |
|------------------------|-----------------|------------|-------|---------------|-----------|
| S2P + Homoskedastic    | 99              | 32       | 50    | 1e-3          | TV        |
| S2P + Heteroskedastic  | 99              | 1024       | 50    | 1e-4          | TV        |
| S2P + Homoskedastic    | 99              | 32       | 50    | 1e-3          | Kettle    |
| S2P + Heteroskedastic  | 99              | 1024       | 50    | 1e-4          | Kettle    |
| S2P + Homoskedastic    | 99              | 32       | 50    | 1e-3          | Washing Machine   |
| S2P + Heteroskedastic  | 99              | 1024       | 50    | 1e-4          | Washing Machine   |

## UK-DALE 

| Models                 | Sequence Length | Batch Size | Epoch | Learning Rate | Appliance |
|------------------------|-----------------|------------|-------|---------------|-----------|
| S2P + Homoskedastic    | 99              | 32       | 50    | 1e-3          | Fridge        |
| S2P + Heteroskedastic  | 99              | 1024       | 50    | 1e-4          | Fridge        |
| S2P + Homoskedastic    | 99              | 32       | 50    | 1e-3          | Kettle    |
| S2P + Heteroskedastic  | 99              | 1024       | 50    | 1e-4          | Kettle    |
| S2P + Homoskedastic    | 99              | 32       | 50    | 1e-3          | Washing Machine   |
| S2P + Heteroskedastic  | 99              | 1024       | 50    | 1e-4          | Washing Machine   |


#### Did you specify all the training details (e.g., data splits, hyperparameters, how they were chosen)?

These houses were specifically chosen on the basis of availability of the appliance data. Hyperparameters have been chosen based on previous literature [https://www.researchgate.net/publication/356356151_Neural_network_approaches_and_dataset_parser_for_NILM_toolkit , https://dl.acm.org/doi/pdf/10.1145/3563357.3564063?casa_token=l6b31X48NgAAAAAA:pUBUiA78iTcaXoAkSFTW4xYmJA6INun2B6FjXFLfVXrLn_B7ddtaIX4vWKcrnmwCPb4EAATTqemj]

