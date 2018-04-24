# Tensorflow_gesture
##如何将.csv文件中的数据解析并进行pc端预测
1：使用Utils/ReadCsv.py文件读取并解析csv文件，执行完成后会将数据以txt文件的形式存至path路径
2：将txt格式的文件切成0.5s的微手势
3：使用切成的微手势利用Predict/gesture_cnn_pb_predict.py文件进行预测

