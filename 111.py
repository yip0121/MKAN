import numpy as np
import matplotlib.pyplot as plt

# 1. 把这里换成你截图里那个长长的文件夹路径
folder_path = '.TimeKAN-main/results/long_term_forecast_battery_soh_20_1_none_TimeKAN_battery_soh_sl20_pl1_dm16_nh4_el2_dl1_df32_fc1_ebtimeF_dtTrue_Exp_0/'

# 2. 读取文件
metrics = np.load(folder_path + 'metrics.npy')
preds = np.load(folder_path + 'pred.npy')
trues = np.load(folder_path + 'true.npy')

# 3. 打印评估指标
# Time-Series 库通常默认指标顺序为: mae, mse, rmse, mape, mspe
print("评估指标 Metrics:", metrics)
print("预测数据形状:", preds.shape)
print("真实数据形状:", trues.shape)

# 4. 画图对比 (因为你的预测长度 pred_len=1，拉平数据画图最直观)
plt.figure(figsize=(12, 5))
plt.plot(trues.flatten(), label='Ground Truth (真实 SOH)', color='blue')
plt.plot(preds.flatten(), label='Prediction (预测 SOH)', color='red', linestyle='--')
plt.title('Battery SOH Prediction vs Ground Truth')
plt.xlabel('Time Steps')
plt.ylabel('SOH')
plt.legend()
plt.show()