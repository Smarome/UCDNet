import pickle

# 打开.pkl文件并读取数据
with open('/root/autodl-tmp/data/TEST23/bevdetv2-nuscenes-test23_infos_val.pkl', 'rb') as pkl_file:
    data = pickle.load(pkl_file)

# 打开.txt文件并将数据写入
with open('TEST23_val—new.txt', 'w') as txt_file:
    txt_file.write(str(data))

print("Pickle文件已成功转换为文本文件。")
