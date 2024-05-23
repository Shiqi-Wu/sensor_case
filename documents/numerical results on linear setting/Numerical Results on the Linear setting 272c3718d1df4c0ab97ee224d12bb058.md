# Numerical Results on the Linear setting

- [Numerical Results on the Linear setting](#numerical-results-on-the-linear-setting)
- [Experiment 1](#experiment-1)
  - [Relative Error](#relative-error)
  - [Config](#config)
  - [Training and Validation Loss](#training-and-validation-loss)
  - [Training](#training)
  - [Testing](#testing)
- [Experiment 2](#experiment-2)
  - [Relative Error](#relative-error-1)
  - [Config](#config-1)
  - [Training and Validation Loss](#training-and-validation-loss-1)
  - [Training](#training-1)
  - [Testing](#testing-1)
- [Experiment 3](#experiment-3)
  - [Relative Error](#relative-error-2)
  - [Config](#config-2)
  - [Training and Validation Loss](#training-and-validation-loss-2)
  - [Training](#training-2)
  - [Testing](#testing-2)
- [Experiment 4](#experiment-4)
  - [Relative Error](#relative-error-3)
  - [Config](#config-3)
  - [Training and Validation Loss](#training-and-validation-loss-3)
  - [Training](#training-3)
  - [Testing](#testing-3)
- [Experiment 5](#experiment-5)
  - [Relative Error](#relative-error-4)
  - [Config](#config-4)
  - [Training and Validation Loss](#training-and-validation-loss-4)
  - [Training](#training-4)
  - [Testing](#testing-4)
- [Experiment 6](#experiment-6)
  - [Relative Error](#relative-error-5)
  - [Config](#config-5)
  - [Training and Validation Loss](#training-and-validation-loss-5)
  - [Training](#training-5)
  - [Testing](#testing-5)

# Experiment 1

## Relative Error

| Model | Sequence Length L | PCA dimension | Dictionary Dimension | Relative Error on Training Data | Relative Error on Testing Data |
| --- | --- | --- | --- | --- | --- |
| Model 1 (Original Koopman Model) | 30 | 4 | 0 | 54.97% | 66.02% |

## Config

```yaml
save_dir: "/home/shiqi/code/Project2-sensor-case/model_combination_Argos/pca_from_formulation/output/experiment_11"
data_dir_list: 
  - "/home/shiqi/code/Project2-sensor-case/model_combination_Argos/data_dir/data_linear_100"
  - "/home/shiqi/code/Project2-sensor-case/model_combination_Argos/data_dir/data_linear_500"
  - "/home/shiqi/code/Project2-sensor-case/model_combination_Argos/data_dir/data_linear_1000"
test_dir_list:
  - "/home/shiqi/code/Project2-sensor-case/model_combination_Argos/data_dir/data_linear_100_test"
  - "/home/shiqi/code/Project2-sensor-case/model_combination_Argos/data_dir/data_linear_500_test"
  - "/home/shiqi/code/Project2-sensor-case/model_combination_Argos/data_dir/data_linear_1000_test"
window_size: 150
predict_num: 30
nu_list:
  - 100
  - 500
  - 1000
pca_dim: 4
dic_model: 0
dd_model: 128
dd_ff: 128
u_model: 32
u_dic_model: 0
u_ff: 128
N_Control: 6
N_State: 6
lr: 0.001
batch_size: 512
num_epochs: 2500
experiment: "linear"
```

## Training and Validation Loss

![Untitled](Numerical%20Results%20on%20the%20Linear%20setting%20272c3718d1df4c0ab97ee224d12bb058/Untitled.png)

## Training

![Untitled](Numerical%20Results%20on%20the%20Linear%20setting%20272c3718d1df4c0ab97ee224d12bb058/Untitled%201.png)

## Testing

![Untitled](Numerical%20Results%20on%20the%20Linear%20setting%20272c3718d1df4c0ab97ee224d12bb058/Untitled%202.png)

# Experiment 2

## Relative Error

| Model | Sequence Length L | PCA dimension | Dictionary Dimension | Relative Error on Training Data | Relative Error on Testing Data |
| --- | --- | --- | --- | --- | --- |
| Model 1 (Original Koopman Model) | 30 | 4 | 40 | 15.17% | 21.83% |

## Config

```yaml
save_dir: "/home/shiqi/code/Project2-sensor-case/model_combination_Argos/pca_from_formulation/output/experiment_15"
data_dir_list: 
  - "/home/shiqi/code/Project2-sensor-case/model_combination_Argos/data_dir/data_linear_100"
  - "/home/shiqi/code/Project2-sensor-case/model_combination_Argos/data_dir/data_linear_500"
  - "/home/shiqi/code/Project2-sensor-case/model_combination_Argos/data_dir/data_linear_1000"
test_dir_list:
  - "/home/shiqi/code/Project2-sensor-case/model_combination_Argos/data_dir/data_linear_100_test"
  - "/home/shiqi/code/Project2-sensor-case/model_combination_Argos/data_dir/data_linear_500_test"
  - "/home/shiqi/code/Project2-sensor-case/model_combination_Argos/data_dir/data_linear_1000_test"
window_size: 150
predict_num: 30
nu_list:
  - 100
  - 500
  - 1000
pca_dim: 4
dic_model: 40
dd_model: 128
dd_ff: 128
u_model: 32
u_dic_model: 0
u_ff: 128
N_Control: 6
N_State: 6
lr: 0.001
batch_size: 512
num_epochs: 2500
experiment: "linear"
```

## Training and Validation Loss

## Training

![Untitled](Numerical%20Results%20on%20the%20Linear%20setting%20272c3718d1df4c0ab97ee224d12bb058/Untitled%203.png)

## Testing

![Untitled](Numerical%20Results%20on%20the%20Linear%20setting%20272c3718d1df4c0ab97ee224d12bb058/Untitled%204.png)

# Experiment 3

## Relative Error

| Model | Sequence Length L | PCA dimension | Dictionary Dimension | Relative Error on Training Data | Relative Error on Testing Data |
| --- | --- | --- | --- | --- | --- |
| Model 2 (Linear Model) | 30 | 4 | 40 | 14.78% | 22.05% |

## Config

```yaml
# Dic With Inputs Without zero data
save_dir: "/home/shiqi/code/Project2-sensor-case/model_combination_Argos/pca_from_formulation/output/experiment_12"
data_dir_list: 
  - "/home/shiqi/code/Project2-sensor-case/model_combination_Argos/data_dir/data_linear_100"
  - "/home/shiqi/code/Project2-sensor-case/model_combination_Argos/data_dir/data_linear_500"
  - "/home/shiqi/code/Project2-sensor-case/model_combination_Argos/data_dir/data_linear_1000"
test_dir_list:
  - "/home/shiqi/code/Project2-sensor-case/model_combination_Argos/data_dir/data_linear_100_test"
  - "/home/shiqi/code/Project2-sensor-case/model_combination_Argos/data_dir/data_linear_500_test"
  - "/home/shiqi/code/Project2-sensor-case/model_combination_Argos/data_dir/data_linear_1000_test"
window_size: 150
predict_num: 30
nu_list:
  - 100
  - 500
  - 1000
pca_dim: 4
dic_model: 40
dd_model: 128
dd_ff: 128
u_model: 32
u_dic_model: 40
u_ff: 128
N_Control: 6
N_State: 6
lr: 0.001
batch_size: 512
num_epochs: 2500
experiment: "linear"
```

## Training and Validation Loss

![Untitled](Numerical%20Results%20on%20the%20Linear%20setting%20272c3718d1df4c0ab97ee224d12bb058/Untitled%205.png)

## Training

![Untitled](Numerical%20Results%20on%20the%20Linear%20setting%20272c3718d1df4c0ab97ee224d12bb058/Untitled%206.png)

## Testing

![Untitled](Numerical%20Results%20on%20the%20Linear%20setting%20272c3718d1df4c0ab97ee224d12bb058/Untitled%207.png)

# Experiment 4

## Relative Error

| Model | Sequence Length L | PCA dimension | Dictionary Dimension | Relative Error on Training Data | Relative Error on Testing Data |
| --- | --- | --- | --- | --- | --- |
| Model 3 (Koopman Model With Inputs in Dictionary) | 30 | 4 | 40 | 14.98% | 21.96% |

## Config

```yaml
# Dic With Inputs Without zero data
save_dir: "/home/shiqi/code/Project2-sensor-case/model_combination_Argos/pca_from_formulation/output/experiment_10"
data_dir_list: 
  - "/home/shiqi/code/Project2-sensor-case/model_combination_Argos/data_dir/data_linear_100"
  - "/home/shiqi/code/Project2-sensor-case/model_combination_Argos/data_dir/data_linear_500"
  - "/home/shiqi/code/Project2-sensor-case/model_combination_Argos/data_dir/data_linear_1000"
test_dir_list:
  - "/home/shiqi/code/Project2-sensor-case/model_combination_Argos/data_dir/data_linear_100_test"
  - "/home/shiqi/code/Project2-sensor-case/model_combination_Argos/data_dir/data_linear_500_test"
  - "/home/shiqi/code/Project2-sensor-case/model_combination_Argos/data_dir/data_linear_1000_test"
window_size: 150
predict_num: 30
nu_list:
  - 100
  - 500
  - 1000
pca_dim: 4
dic_model: 40
dd_model: 128
dd_ff: 128
u_model: 32
u_ff: 128
N_Control: 6
N_State: 6
lr: 0.001
batch_size: 512
num_epochs: 2500
experiment: "linear"
experiment: "DicWithInputs"
```

## Training and Validation Loss

![Untitled](Numerical%20Results%20on%20the%20Linear%20setting%20272c3718d1df4c0ab97ee224d12bb058/Untitled%208.png)

## Training

![Untitled](Numerical%20Results%20on%20the%20Linear%20setting%20272c3718d1df4c0ab97ee224d12bb058/Untitled%209.png)

## Testing

![Untitled](Numerical%20Results%20on%20the%20Linear%20setting%20272c3718d1df4c0ab97ee224d12bb058/Untitled%2010.png)

# Experiment 5

## Relative Error

| Model | Sequence Length L | PCA dimension | Dictionary Dimension | Relative Error on Training Data | Relative Error on Testing Data |
| --- | --- | --- | --- | --- | --- |
| Model 3 (Koopman Model With Inputs in Dictionary) | 10 | 4 | 40 | 17.69% | 24.92% |

## Config

```yaml
# Dic With Inputs Without zero data
save_dir: "/home/shiqi/code/Project2-sensor-case/model_combination_Argos/pca_from_formulation/output/experiment_14"
data_dir_list: 
  - "/home/shiqi/code/Project2-sensor-case/model_combination_Argos/data_dir/data_linear_100"
  - "/home/shiqi/code/Project2-sensor-case/model_combination_Argos/data_dir/data_linear_500"
  - "/home/shiqi/code/Project2-sensor-case/model_combination_Argos/data_dir/data_linear_1000"
test_dir_list:
  - "/home/shiqi/code/Project2-sensor-case/model_combination_Argos/data_dir/data_linear_100_test"
  - "/home/shiqi/code/Project2-sensor-case/model_combination_Argos/data_dir/data_linear_500_test"
  - "/home/shiqi/code/Project2-sensor-case/model_combination_Argos/data_dir/data_linear_1000_test"
window_size: 150
predict_num: 10
nu_list:
  - 100
  - 500
  - 1000
pca_dim: 4
dic_model: 40
dd_model: 128
dd_ff: 128
u_model: 32
u_dic_model: 40
u_ff: 128
N_Control: 6
N_State: 6
lr: 0.001
batch_size: 512
num_epochs: 2500
experiment: "DicWithInputs"
```

## Training and Validation Loss

![Untitled](Numerical%20Results%20on%20the%20Linear%20setting%20272c3718d1df4c0ab97ee224d12bb058/Untitled%2011.png)

## Training

![Untitled](Numerical%20Results%20on%20the%20Linear%20setting%20272c3718d1df4c0ab97ee224d12bb058/Untitled%2012.png)

## Testing

![Untitled](Numerical%20Results%20on%20the%20Linear%20setting%20272c3718d1df4c0ab97ee224d12bb058/Untitled%2013.png)

# Experiment 6

## Relative Error

| Model | Sequence Length L | PCA dimension | Dictionary Dimension | Relative Error on Training Data | Relative Error on Testing Data |
| --- | --- | --- | --- | --- | --- |
| Model 4 (Koopamn Model With Inputs In Operator) | 10 | 4 | 40 | 16.07% | 18.86% |

## Config

```yaml
# Dic With Inputs Without zero data
save_dir: "/home/shiqi/code/Project2-sensor-case/model_combination_Argos/pca_from_formulation/output/experiment_13"
data_dir_list: 
  - "/home/shiqi/code/Project2-sensor-case/model_combination_Argos/data_dir/data_linear_100"
  - "/home/shiqi/code/Project2-sensor-case/model_combination_Argos/data_dir/data_linear_500"
  - "/home/shiqi/code/Project2-sensor-case/model_combination_Argos/data_dir/data_linear_1000"
test_dir_list:
  - "/home/shiqi/code/Project2-sensor-case/model_combination_Argos/data_dir/data_linear_100_test"
  - "/home/shiqi/code/Project2-sensor-case/model_combination_Argos/data_dir/data_linear_500_test"
  - "/home/shiqi/code/Project2-sensor-case/model_combination_Argos/data_dir/data_linear_1000_test"
window_size: 150
predict_num: 10
nu_list:
  - 100
  - 500
  - 1000
pca_dim: 4
dic_model: 40
dd_model: 128
dd_ff: 128
u_model: 32
u_dic_model: 40
u_ff: 128
N_Control: 6
N_State: 6
lr: 0.001
batch_size: 512
num_epochs: 2500
experiment: "MatrixWithInputs"
```

## Training and Validation Loss

![Untitled](Numerical%20Results%20on%20the%20Linear%20setting%20272c3718d1df4c0ab97ee224d12bb058/Untitled%2014.png)

## Training

![Untitled](Numerical%20Results%20on%20the%20Linear%20setting%20272c3718d1df4c0ab97ee224d12bb058/Untitled%2015.png)

## Testing

![Untitled](Numerical%20Results%20on%20the%20Linear%20setting%20272c3718d1df4c0ab97ee224d12bb058/Untitled%2016.png)