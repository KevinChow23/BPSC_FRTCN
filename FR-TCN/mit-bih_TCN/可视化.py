import wfdb
import matplotlib.pyplot as plt

# 指定心电图记录的路径
# 这可以是本地路径，也可以是数据库中的记录名，如 'mitdb/100'
record_path = 'mit-bih-arrhythmia-database-1.0.0/100'

# 读取记录
# sampfrom和sampto参数分别用来指定读取信号的开始和结束位置（可选）
record = wfdb.rdrecord(record_path, sampfrom=0, sampto=1000)

# 可视化ECG信号
# 这会画出所有可用的导联，可以通过设置channels参数来选择特定的导联
wfdb.plot_wfdb(record=record, title='ECG Signal', time_units='seconds')

# 如果只想画出ECG信号而不包含注释，可以使用plt.plot直接绘制
# 获取信号和时间向量
ecg_signal = record.p_signal
time = record.fs * record.sig_len  # fs是采样频率，sig_len是信号长度

# 绘制信号的第一个导联，例如
plt.figure(figsize=(10, 4))
plt.plot(ecg_signal[:, 0])
plt.title('ECG Lead I')
plt.xlabel('Time [samples]')
plt.ylabel('Amplitude [mV]')
plt.show()
