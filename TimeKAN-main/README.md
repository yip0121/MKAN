
<div align="center">
  <h2><b> (ICLR 2025) TimeKAN: KAN-based Frequency Decomposition Learning Architecture for Long-term Time Series Forecasting🚀 </b></h2>
</div>

### This is an offical implementation of "TimeKAN: KAN-based Frequency Decomposition Learning Architecture for Long-term Time Series Forecasting" 

## Overall Architecture
<p align="center">
<img src="./Figure/TimeKAN.jpg"  alt="" align=center />
</p>


## Results
<p align="center">
<img src="./Figure/result.png"  alt="" align=center />
</p>

## Getting Started
1. Install requirements. ```pip install -r requirements.txt```

2. Download data. You can download all the datasets from [Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy). 

3. Training. All the scripts are in the directory ```.scripts```.  If you want to obtain the results of **input-96-predict-96** on the Weather dataset, you can run the following command:
```
sh scripts/long_term_forecast/Weather/weather_96.sh
```


## Battery Dataset Placement (Custom)
- Put battery files under `./dataset/battery/` (not under `./data_provider/`).
- The training loader reads CSV files (`pd.read_csv`), so convert battery Excel files to CSV before training.
- Example converted filename:
  - `battery_36Ah_70W_65W_1551.csv`
- Use the following arguments for training:
  - `--root_path ./dataset/battery/`
  - `--data_path battery_36Ah_70W_65W_1551.csv`


## Acknowledgement

We sincerely appreciate the following github repo very much for the valuable code base and datasets:

https://github.com/cure-lab/LTSF-Linear

https://github.com/kwuking/TimeMixer

https://github.com/thuml/Time-Series-Library

https://github.com/ts-kim/RevIN

https://github.com/SynodicMonth/ChebyKAN


## Citation

If you find this repository useful for your work, please consider citing it as follows:

```BibTeX
@inproceedings{
  huang2025timekan,
  title={Time{KAN}: {KAN}-based Frequency Decomposition Learning Architecture for Long-term Time Series Forecasting},
  author={Songtao Huang and Zhen Zhao and Can Li and LEI BAI},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=wTLc79YNbh}
}
```
