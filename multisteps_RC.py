from scipy.optimize import curve_fit
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import numpy as np

import pandas as pd

# 读取节点坐标表,转换为DataFrame
coords = pd.read_excel(r"观测点坐标.xlsx")  
coords_df = pd.DataFrame(coords)


# 合并三个速度DataFrame
v_df_1 = pd.read_excel(r"G:\Emist\Flow_surrogate\factory_with_plant_units_and_leds_simplified\led修正无蒸腾制冷训练v.xlsx"#"G:\Emist\Flow_surrogate\factory3\无蒸腾十天\v_magn_1.xlsx"
                       )
v_df_2 = pd.read_excel(r"G:\Emist\Flow_surrogate\factory_with_plant_units_and_leds_simplified\led修正无蒸腾制冷测试v.xlsx"
                       )

t_df_1 = pd.read_excel(r"G:\Emist\Flow_surrogate\factory_with_plant_units_and_leds_simplified\led修正无蒸腾制冷训练t.xlsx"#"G:\Emist\Flow_surrogate\factory3\无蒸腾十天\static_t_1.xlsx"
                       )
t_df_2 = pd.read_excel(r"G:\Emist\Flow_surrogate\factory_with_plant_units_and_leds_simplified\led修正无蒸腾制冷测试t.xlsx"
                       )

param_train_table = pd.read_json(r'G:\Emist\Flow_surrogate\param_train_table_0927.json')

v_df_1 = v_df_1[:282]
print(len(v_df_1))

t_df_1 = t_df_1[:282]
print(len(t_df_1))

v_df_2 = v_df_2[:282]
print(len(v_df_2))

t_df_2 = t_df_2[:282]
print(len(t_df_2))

ac_T = t_df_1['a1']#[v[1] for v in param_train_table['day 1']['data'][2]][1:]
ac_v = v_df_1['a1']#[v[1] for v in param_train_table['day 1']['data'][3]][1:]

ac_T_2 = t_df_2['a1']#[v[1] for v in param_train_table['day 2']['data'][2]][1:]
ac_v_2 = v_df_2['a1']#[v[1] for v in param_train_table['day 2']['data'][3]][1:]

expanded_ac_T = [0] * len(ac_T) * 6 #半小时到五分钟；六倍大小数组
expanded_ac_v = [0] * len(ac_v) * 6

expanded_ac_T_2 = [0] * len(ac_T) * 6 #半小时到五分钟；六倍大小数组
expanded_ac_v_2 = [0] * len(ac_v) * 6

for i in range(len(ac_T)):
    T = ac_T[i]
    v = ac_v[i]
    T_2 = ac_T_2[i]
    v_2 = ac_v_2[i]

    start_index = i * 6
    for j in range(6):
        expanded_ac_T[start_index + j] = T
        expanded_ac_v[start_index + j] = v
        expanded_ac_T_2[start_index + j] = T_2
        expanded_ac_v_2[start_index + j] = v_2

ac_T = expanded_ac_T#[:-20]
ac_v = expanded_ac_v#[:-20]
ac_T_2 = expanded_ac_T_2#[:-20]
ac_v_2 = expanded_ac_v_2#[:-20]

print(len(ac_T))

leds = []#W/m2
for i in range(len(ac_T)):
    if i * 60 * 5 < 8 * 60 * 60: #8小时暗期
        leds.append(0)
    else:
        leds.append(1)

leds_2 = []#W/m2
for i in range(len(ac_T_2)):
    if i * 60 * 5 < 8 * 60 * 60: #8小时暗期
        leds_2.append(0)
    else:
        leds_2.append(1)

return_T = [t for t in t_df_1['return_node']][1:-1] 
  
ac_Q = [(T-rT+273.15)*v*0.6*1.2*1.005 for T,v,rT in zip(ac_T,ac_v,return_T)]#m/s,m2, K, kg/m3, kJ/kg.K #kW
combine_coords_df = {}
for i in range(len(coords_df['编号'])):
    combine_coords_df[coords_df['编号'][i]] = [(coords_df['x'][i],coords_df['y'][i],coords_df['z'][i])]
combine_coords_df = pd.DataFrame(combine_coords_df)

import json
with open(r'G:\Emist\Flow_surrogate\classes.json', 'r') as f:
    classes = json.load(f)

nodes_dict = classes


def rmse(y_true,y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    return rmse

def average_by_groups(array, group_size):
    averaged_array = []
    n = len(array)
    
    for i in range(0, n, group_size):
        group = array[i:i+group_size]
        group_average = sum(group) / len(group)
        averaged_array.append(group_average)
    
    return averaged_array

def rolling_average(array, window_size): #得到某个时间段的半小时的平均温度；否则static温度有些吃不准
    #以五分钟为窗口长度的半小时平均温度曲线
    averaged_array = []
    n = len(array)

    # 计算第一个窗口的平均值
    window_sum = sum(array[:window_size])
    averaged_array.append(window_sum / window_size)

    # 滚动窗口计算平均值
    for i in range(1, n - window_size + 1):
        window_sum = window_sum - array[i - 1] + array[i + window_size - 1]
        averaged_array.append(window_sum / window_size)

    return averaged_array

class multisteps_RC(): #对流+辐射传热RC模型
    #代表单独一个节点的临时情况；计算的结果，拟合的参数在训练后都要上交到一个rc管理类里面。应用的时候它相当于临时调用的容器，rc管理类把对应节点又给到它让它运行
    def __init__(self):
         #s
        self.steps = 1 #1就是五分钟
        self.step_lenght = self.steps * 60
        self.steps_list = [1, 2, 3, 6, 10]  #步长列表；单位五分钟; 5,10,15,30,60
        self.rc_inputs_dict = {} #真实输入
        self.rc_outputs_dict = {} #真实输出
        self.rc_test_outputs_dict = {} #拟合输出

        self.validate_pred = {} #测试集预测输出
        self.validate_true = {} #测试集真实输出

        self.best_mse_dict = {}
        self.best_params_dict = {}
        self.best_params_retrained_dict = {}
        self.validate_performance_dict = {}
        self.object = 'a4'
        self.class_now = 'class2'
        
        self.count_run_times = 0
        
        self.vel_validate_performance_dict = {} #风速预测的性能指标
        self.vel_black_box_models = {}
        self.classes = classes
        
    def RC_model(self, inputs, *params):
        #print(params)
        C = params[0] #热容单位：kJ/K
        a_self = params[1]
        b_self = params[2]
        c_self = params[3]
        d_self = params[4]        

        a_nb = params[5]
        b_nb = params[6]
        c_nb = params[7]
        d_nb = params[8]

        a_rad = params[9]
        b_rad = params[10]
        c_rad = params[11]
        d_rad = params[12]

        time_param = params[13]

        led_W = inputs[0] #kW
        ac_T = inputs[1]
        ac_v = inputs[2]
        t_self = inputs[3]
        #v_self = inputs[4]
        t_nb = inputs[4]
        #v_nb = inputs[6]

        #R_self = a_self + b_self * v_self + c_self * v_self**2 #+ d_self * v_self**3 
        # #热阻单位：K/kW
        R_self =  a_self + b_self * ac_v + c_self * ac_v **2
        #R_nb = a_nb + b_nb * v_nb + c_nb * v_nb**2 #+ d_nb * v_nb**3
        #R_nb = d_nb
        #Kt 时间系数：300秒
        R_nb = a_nb + b_nb * ac_v + c_nb * ac_v**2
        
        T_rad = a_rad + b_rad * led_W * 4608 + c_rad * (led_W*4608)**2 #+ d_rad * led_W**3 #瞎几把拟合led辐射对空气温度的影响

        Q_cooling = ac_v * 0.6 * 1.2 * 1.005 * (ac_T - t_self) #kW #时空相对制冷量；某一个方位的节点的上一刻的

        new_T = (t_self - t_nb) * np.exp(-time_param / (C * R_nb)) + Q_cooling * R_self + T_rad #t_self + (t_self - t_nb) * np.exp(-time_param / (C * R_nb)) + Q_cooling * R_self + T_rad 

        return new_T
    
    def data_treat(self,status,t_df_1,v_df_1,leds,ac_T,ac_v): #status表明这个是训练还是应用
        #先按照要得步长重新处理，然后再组输入和输出
        # 先取前3个相似节点 
        #print(self.object)
        if status == "train": #train的话就要重新看谁离谁最近
            object_r2_dict = {k: r2_score(t_df_1[self.object], t_df_1[k]) for k in self.classes[self.class_now]}
            
            if self.class_now != 'class1':             
                object_r2_dict = sorted(object_r2_dict.items(), key=lambda x: x[1], reverse=True)[:3]
                try:
                    self.neighbor_nodes = [object_r2_dict[0][0],object_r2_dict[1][0],object_r2_dict[2][0]]
                except:
                    self.neighbor_nodes = [object_r2_dict[0][0],object_r2_dict[1][0]]
            else:
                object_r2_dict = sorted(object_r2_dict.items(), key=lambda x: x[1], reverse=True)[0]
                self.neighbor_nodes = [object_r2_dict[0]]
                
        elif status == 'apply':#apply就跳过不需要再设定一遍了 #apply的时候rc_manage 会分配已知的最近的三个节点
            pass

        # 然后根据步长调整
        av = self.steps
        av_ld = average_by_groups(leds,av)
        av_ac_T = average_by_groups(ac_T,av)
        av_ac_v = average_by_groups(ac_v,av)
        #print(len(leds))
        #print(len(av_ld))
        involved_nodes = classes[self.class_now]
        involved_nodes.append(self.object)
        av_t_df_1 = {k: average_by_groups(t_df_1[k],av) for k in self.classes[self.class_now]}
        av_v_df_1 = {k: average_by_groups(v_df_1[k],av) for k in self.classes[self.class_now]}

        av_t_df_1[self.object] = average_by_groups(t_df_1[self.object],av)
        av_v_df_1[self.object] = average_by_groups(v_df_1[self.object],av)
        #print(len(t_df_1[self.object]))
        #print(len(av_t_df_1[self.object]))
        
        if self.class_now != 'class1':
            try:
                av_t_nb_mean = (np.array(av_t_df_1[self.neighbor_nodes[0]]) #因为空间上是滚动预测；一步一步来，所以相邻节点的温度直接取前面的
                + np.array(av_t_df_1[self.neighbor_nodes[1]])
                + np.array(av_t_df_1[self.neighbor_nodes[2]]))/3
                
                av_v_nb_mean = (np.array(av_v_df_1[self.neighbor_nodes[0]])
                + np.array(av_v_df_1[self.neighbor_nodes[1]])
                + np.array(av_v_df_1[self.neighbor_nodes[2]]))/3
            except:
                av_t_nb_mean = (np.array(av_t_df_1[self.neighbor_nodes[0]]) #因为空间上是滚动预测；一步一步来，所以相邻节点的温度直接取前面的
                + np.array(av_t_df_1[self.neighbor_nodes[1]])
                )/3
                
                av_v_nb_mean = (np.array(av_v_df_1[self.neighbor_nodes[0]])
                + np.array(av_v_df_1[self.neighbor_nodes[1]])
                )/3
            
        else:
            av_t_nb_mean = np.array(av_t_df_1[self.neighbor_nodes[0]])
            av_v_nb_mean = np.array(av_v_df_1[self.neighbor_nodes[0]])

        self.rc_inputs = [ #临时的
            av_ld[1:],
            av_ac_T[1:],#考量的是空调输出影响节点的即时性 #用现在的，因为前面的空调控制产生的影响已经有t_nb带来
            av_ac_v[1:],
            av_t_df_1[self.object][:-1],
            #av_v_df_1[self.object][1:-1-5],
            av_t_nb_mean[1:],#隔壁的就是现在的
            #av_v_nb_mean[:-5]
        ]
        
        for l in self.rc_inputs:
            print(len(l))
        self.rc_inputs = np.array(self.rc_inputs)

        self.rc_outputs = np.array(av_t_df_1[self.object][1:]) #- np.array(av_t_df_1[self.object][:-1])

        self.rc_outputs = np.array(self.rc_outputs)
        
        #数据插值试试用来增加数据量
        #self.rc_inputs,self.rc_outputs = self.interpolate_multi_dimensional_data(np.transpose(self.rc_inputs),np.transpose(self.rc_outputs),2000)
        #self.rc_inputs, self.rc_outputs = np.transpose(self.rc_inputs),np.transpose(self.rc_outputs)#再转回来
    def stupid_data_treat(self,status,t_df_1,v_df_1,leds,ac_T,ac_v):
        
        # 然后根据步长调整
        av = self.steps
        av_ld = average_by_groups(leds,av)
        av_ac_T = average_by_groups(ac_T,av)
        av_ac_v = average_by_groups(ac_v,av)

        av_t_df_1 = {k: average_by_groups(t_df_1[k],av) for k in list(t_df_1.keys())[1:-1]} #去掉首尾不要的列

        self.rc_inputs = [ #临时的
            av_ld[1:-1-5],
            av_ac_T[:-2-5],
            av_ac_v[:-2-5],
        ]
        for key in list(t_df_1.keys())[1:-1]:
            self.rc_inputs.append(av_t_df_1[key][1:-1])
        
        for l in self.rc_inputs:
            print(len(l))
        self.rc_inputs = np.array(self.rc_inputs)

        self.rc_outputs = av_t_df_1[self.object][2:]

        self.rc_outputs = np.array(self.rc_outputs)
        
    def data_treat_all(self,t_df_1,v_df_1,leds,ac_T,ac_v,t_df_2,v_df_2,leds_2,ac_T_2,ac_v_2):
        #方法：气流路径+黑箱
        #输入：气流路径前一步的温度+设备状态
        #作用：训练+验证
        self.data_treat("train",t_df_1,v_df_1,leds,ac_T,ac_v)
        self.train_X = self.rc_inputs
        for l in self.train_X:
            print(len(l))
        self.train_y = self.rc_outputs
        print(len(self.train_y))
        self.data_treat("apply",t_df_2,v_df_2,leds_2,ac_T_2,ac_v_2)
        self.test_X = self.rc_inputs
        self.test_y = self.rc_outputs
        
        return self.black_box_models_build_train_and_validate(self.train_X,self.train_y,self.test_X,self.test_y)
    
    def only_ac_data_treat_all(self,t_df_1,v_df_1,leds,ac_T,ac_v,t_df_2,v_df_2,leds_2,ac_T_2,ac_v_2):
        #方法：黑箱
        #输入：设备状态
        #作用：训练+验证
        self.data_treat("train",t_df_1,v_df_1,leds,ac_T,ac_v)
        self.train_X = self.rc_inputs[:3]
        #for l in self.train_X:
        #    print(len(l))
        self.train_y = self.rc_outputs
        #print(len(self.train_y))
        self.data_treat("apply",t_df_2,v_df_2,leds_2,ac_T_2,ac_v_2)
        self.test_X = self.rc_inputs[:3]
        self.test_y = self.rc_outputs
        
        return self.black_box_models_build_train_and_validate(self.train_X,self.train_y,self.test_X,self.test_y)

    def direct_data_treat_all(self,t_df_1,v_df_1,leds,ac_T,ac_v,t_df_2,v_df_2,leds_2,ac_T_2,ac_v_2):
        #方法：黑箱
        #输入：所有节点的上一时刻温度+设备状态
        #作用：训练+验证
        
        self.stupid_data_treat("train",t_df_1,v_df_1,leds,ac_T,ac_v)
        self.train_X = self.rc_inputs
        self.train_y = self.rc_outputs
        self.stupid_data_treat("apply",t_df_2,v_df_2,leds_2,ac_T_2,ac_v_2)
        self.test_X = self.rc_inputs
        self.test_y = self.rc_outputs
        
        return self.black_box_models_build_train_and_validate(self.train_X,self.train_y,self.test_X,self.test_y)
        
    def vel_data_treat(self, status, v_df_1, ac_v): #风速预测所需数据预处理
        if status == "train":
            object_r2_dict = {k: r2_score(v_df_1[self.object], v_df_1[k]) for k in classes[self.class_now]}
            if self.class_now != "class1":
                object_r2_dict = sorted(object_r2_dict.items(), key=lambda x: x[1], reverse=True)[:3]
                self.vel_neighbor_nodes = [object_r2_dict[0][0],object_r2_dict[1][0],object_r2_dict[2][0]]
                
            else:
                object_r2_dict = sorted(object_r2_dict.items(), key=lambda x: x[1], reverse=True)[0]
                self.vel_neighbor_nodes = [object_r2_dict[0]]
                
        elif status == "apply": #apply的时候rc_manage 会分配已知的最近的三个节点
            pass
        
        av = self.steps
        av_ac_v = average_by_groups(ac_v,av)
        av_v_df_1 = {k: average_by_groups(v_df_1[k],av) for k in classes[self.class_now]}
        av_v_df_1[self.object] = average_by_groups(v_df_1[self.object],av)
        
        if self.class_now != 'class1':
            av_v_nb_mean = (np.array(av_v_df_1[self.vel_neighbor_nodes[0]][2:])
            + np.array(av_v_df_1[self.vel_neighbor_nodes[1]][2:])
            + np.array(av_v_df_1[self.vel_neighbor_nodes[2]][2:]))/3
        else:
            av_v_nb_mean = np.array(av_v_df_1[self.vel_neighbor_nodes[0]][2:])
            
        self.vel_inputs = np.array(
            [
            av_ac_v[:-2],
            av_v_df_1[self.object][1:-1],
            av_v_nb_mean,
            av_v_nb_mean,
            av_v_nb_mean,
            av_v_nb_mean,
            av_ac_v[:-2],
            av_v_df_1[self.object][1:-1],
        ])
        
        for ngh in self.vel_neighbor_nodes:
            list(self.vel_inputs).append(av_v_df_1[ngh][:-2])
        
        self.vel_inputs = np.array(self.vel_inputs)
        
        self.vel_outputs = np.array(av_v_df_1[self.object][2:])
        
    def objective_function(self,params):
        
        calculated_outputs = self.RC_model(self.rc_inputs, params)
        mse = np.mean((calculated_outputs - self.rc_outputs) ** 2)
        return mse
    
    def train_model(self): #温度rc #用minimize
        # Provide an initial guess for the params
        initial_params = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 600]
        # Run the optimization
        result = minimize(self.objective_function, initial_params, method='Nelder-Mead')
        # Get the optimized params
        optimized_params = result.x
        best_mse = result.fun

        return optimized_params#, best_mse

    def train_model_2(self): #风速 #用curve_fit
        initial_params = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 600]
        params, _ = curve_fit(self.RC_model, self.rc_inputs, self.rc_outputs, p0=initial_params)

        return params

    def interpolate_multi_dimensional_data(self, X, y, num_points=100):
        """
        Interpolate multi-dimensional time series data (X) and one-dimensional target data (y)
        using cubic spline interpolation.

        Parameters:
        X (ndarray): Multi-dimensional time series data with shape (num_samples, num_features).
        y (ndarray): One-dimensional target data with shape (num_samples,).
        num_points (int): Number of interpolated points for each feature dimension.

        Returns:
        ndarray: Interpolated multi-dimensional time series data with shape (num_points, num_features).
        ndarray: Interpolated one-dimensional target data with shape (num_points,).
        """
        import numpy as np
        from scipy.interpolate import CubicSpline
        
        interpolated_features = []

        for feature_idx in range(X.shape[1]):
            x = np.arange(X.shape[0])
            y_feature = X[:, feature_idx]

            # Apply cubic spline interpolation to feature
            cs = CubicSpline(x, y_feature)
            interpolated_x = np.linspace(x[0], x[-1], num_points)
            interpolated_y_feature = cs(interpolated_x)

            interpolated_features.append(interpolated_y_feature)

        interpolated_data = np.vstack(interpolated_features).T  # Shape (num_points, num_features)

        # Apply cubic spline interpolation to target (y)
        cs_y = CubicSpline(x, y)
        interpolated_y = cs_y(interpolated_x)

        return interpolated_data, interpolated_y

    def black_box_models_build_train_and_validate(self,previous_step_vals,current_node_vals, test_set_X, test_set_y):
        #分别建立线性回归，SVR，SVM与LSTM模型
        from sklearn.linear_model import LinearRegression
        from sklearn.svm import SVR
        from keras.models import Sequential
        from keras.layers import LSTM, Dense,Dropout,Bidirectional
        
        X_train = np.transpose(previous_step_vals)
        y_train = np.transpose(current_node_vals)
        X_test = np.transpose(test_set_X)
        test_set_y = np.transpose(test_set_y)
        
        #X_train,y_train = self.interpolate_multi_dimensional_data(X_train,y_train,1000)
        #X_test, test_set_y = self.interpolate_multi_dimensional_data(X_test,test_set_y,1000)
        
        #print(X_train.shape)
        #print(y_train.shape)
        print("Linear Regression")
        self.linear_reg_model = LinearRegression()
        #print(X_train)
        #print(y_train)
        self.linear_reg_model.fit(X_train, y_train)
        print("SVR")
        self.svr_model = SVR()
        self.svr_model.fit(X_train, y_train)
        print("SVM")
        self.svm_model = SVR(kernel="rbf")
        self.svm_model.fit(X_train,y_train)
        print("LSTM")
        X_train = np.reshape(X_train,(X_train.shape[0],1,X_train.shape[1]))
        self.lstm_model = Sequential()
        self.lstm_model.add(Bidirectional(LSTM(units=256,
                                               input_shape=(1,X_train.shape[2])
                                               )))
        #self.lstm_model.add(LSTM(units=64,return_sequences=True))
        #self.lstm_model.add(LSTM(units=64,return_sequences=False))
        self.lstm_model.add(Dropout(0.2))
        self.lstm_model.add(Dense(units=1))
        self.lstm_model.compile(loss="mean_squared_error",optimizer='adam')
        self.lstm_model.fit(X_train,y_train,epochs=200,batch_size=128,verbose=0)       
        
        
        self.lr_y_pred = self.linear_reg_model.predict(X_test)
        self.svr_y_pred = self.svr_model.predict(X_test)
        self.svm_y_pred = self.svm_model.predict(X_test)
        X_test = np.reshape(X_test,(X_test.shape[0],1,X_test.shape[1]))
        self.lstm_y_pred = self.lstm_model.predict(X_test)
        
        self.lr_performance = (r2_score(test_set_y,self.lr_y_pred),rmse(test_set_y,self.lr_y_pred),mean_absolute_percentage_error(test_set_y,self.lr_y_pred))
        self.svr_performance = (r2_score(test_set_y,self.svr_y_pred),rmse(test_set_y,self.svr_y_pred),mean_absolute_percentage_error(test_set_y,self.svr_y_pred))
        self.svm_performance = (r2_score(test_set_y,self.svm_y_pred),rmse(test_set_y,self.svm_y_pred),mean_absolute_percentage_error(test_set_y,self.svm_y_pred))
        self.lstm_performance = (r2_score(test_set_y,self.lstm_y_pred),rmse(test_set_y,self.lstm_y_pred),mean_absolute_percentage_error(test_set_y,self.lstm_y_pred))
        
        return self.lr_performance,self.svr_performance, self.svm_performance, self.lstm_performance

    def black_box_models_build_and_train(self,previous_step_vals,current_node_vals):
        #与前面一个相比是拿训练集又测了一遍然后自定义提取性能指标
        #分别建立线性回归，SVR，SVM与LSTM模型
        from sklearn.linear_model import LinearRegression
        from sklearn.svm import SVR
        from keras.models import Sequential
        from keras.layers import LSTM, Dense, Dropout
        from keras.regularizers import l2
        
        X_train = np.transpose(previous_step_vals)
        y_train = np.transpose(current_node_vals)
        print("Linear Regression")
        #lens = [len(l) for l in X_train]
        #print(lens)
        #print(len(y_train))
        self.linear_reg_model = LinearRegression()
        self.linear_reg_model.fit(X_train, y_train)
        print("SVR")
        self.svr_model = SVR()
        self.svr_model.fit(X_train, y_train)
        print("SVM")
        self.svm_model = SVR(kernel="rbf")
        self.svm_model.fit(X_train,y_train)
        print("LSTM")
        X_train = np.reshape(X_train,(X_train.shape[0],1,X_train.shape[1]))
        self.lstm_model = Sequential()
        self.lstm_model.add(LSTM(units=128,input_shape=(1,X_train.shape[2])))
        self.lstm_model.add(Dense(units=1))
        #self.lstm_model.add(LSTM(units=64))
        #self.lstm_model.add(Dropout(0.2))
        self.lstm_model.compile(loss="mean_squared_error",optimizer='adam')
        self.lstm_model.fit(X_train,y_train,epochs=200,batch_size=64,verbose=0)       
        
        """
        X_test = previous_step_vals[:,:]
        test_set_y = current_node_vals
        
        self.lr_y_pred = self.linear_reg_model.predict(X_test)
        self.svr_y_pred = self.svr_model.predict(X_test)
        self.svm_y_pred = self.svm_model.predict(X_test)
        X_test = np.reshape(X_test,(X_test.shape[0],1,X_test.shape[1]))
        self.lstm_y_pred = self.lstm_model.predict(X_test)
        
        self.lr_performance = (r2_score(test_set_y,self.lr_y_pred),rmse(test_set_y,self.lr_y_pred),mean_absolute_percentage_error(test_set_y,self.lr_y_pred))
        self.svr_performance = (r2_score(test_set_y,self.svr_y_pred),rmse(test_set_y,self.svr_y_pred),mean_absolute_percentage_error(test_set_y,self.svr_y_pred))
        self.svm_performance = (r2_score(test_set_y,self.svm_y_pred),rmse(test_set_y,self.svm_y_pred),mean_absolute_percentage_error(test_set_y,self.svm_y_pred))
        self.lstm_performance = (r2_score(test_set_y,self.lstm_y_pred),rmse(test_set_y,self.lstm_y_pred),mean_absolute_percentage_error(test_set_y,self.lstm_y_pred))
        return self.lr_performance, self.svr_performance, self.svm_performance, self.lstm_performance

        """
        
        
    def train_one(self,t_df_1,v_df_1,leds,ac_T,ac_v):

        self.data_treat("train",t_df_1,v_df_1,leds,ac_T,ac_v)
        
        #X_train,y_train = self.interpolate_multi_dimensional_data(X_train,y_train,1000)
        #X_test, test_set_y = self.interpolate_multi_dimensional_data(X_test,test_set_y,1000)
        
        optimized_params = self.train_model_2()

        return optimized_params#, best_mse
    
    #def 
    
    def vel_train_one(self,v_df_1,ac_v):
        
        self.vel_data_treat("train",v_df_1,ac_v)
        self.train_inputs = self.vel_inputs
        self.train_outputs = self.vel_outputs
        
        self.black_box_models_build_and_train(self.train_inputs,self.train_outputs)

    def vel_validate(self, v_df_2, ac_v_2):
        
        self.vel_data_treat("apply", v_df_2, ac_v_2)
        self.validate_inputs = np.transpose(self.vel_inputs)
        self.validate_outputs = self.vel_outputs
        
        self.lr_y_pred = self.linear_reg_model.predict(self.validate_inputs)
        self.svr_y_pred = self.svr_model.predict(self.validate_inputs)
        self.svm_y_pred = self.svm_model.predict(self.validate_inputs)
        
        self.validate_inputs = np.reshape(self.validate_inputs,(self.validate_inputs.shape[0],1,self.validate_inputs.shape[1]))
        self.lstm_y_pred = self.lstm_model.predict(self.validate_inputs)
        
        self.lr_performance = (r2_score(self.validate_outputs,self.lr_y_pred),rmse(self.validate_outputs,self.lr_y_pred),mean_absolute_percentage_error(self.validate_outputs,self.lr_y_pred))
        self.svr_performance = (r2_score(self.validate_outputs,self.svr_y_pred),rmse(self.validate_outputs,self.svr_y_pred),mean_absolute_percentage_error(self.validate_outputs,self.svr_y_pred))
        self.svm_performance = (r2_score(self.validate_outputs,self.svm_y_pred),rmse(self.validate_outputs,self.svm_y_pred),mean_absolute_percentage_error(self.validate_outputs,self.svm_y_pred))
        self.lstm_performance = (r2_score(self.validate_outputs,self.lstm_y_pred),rmse(self.validate_outputs,self.lstm_y_pred),mean_absolute_percentage_error(self.validate_outputs,self.lstm_y_pred))
        
        return self.lr_performance, self.svr_performance, self.svm_performance, self.lstm_performance
        
    def validate(self,t_df_2,v_df_2,leds_2,ac_T_2,ac_v_2):
        self.data_treat("apply",t_df_2,v_df_2,leds_2,ac_T_2,ac_v_2)

        self.test_outputs = self.RC_model(self.rc_inputs, *self.optimized_params)

        investigate = (
            rmse(self.rc_outputs, self.test_outputs), 
            mean_absolute_percentage_error(self.rc_outputs, self.test_outputs), 
            mean_squared_error(self.rc_outputs, self.test_outputs), 
            r2_score(self.rc_outputs, self.test_outputs)
        )

        return investigate

    def train_all(self,t_df_1,v_df_1,leds,ac_T,ac_v):

        for step in self.steps_list:
            self.steps = step
            self.step_lenght = self.steps * 60
            optimized_params = self.train_one(t_df_1,v_df_1,leds,ac_T,ac_v)
            self.best_params_dict[step] = optimized_params
            self.rc_inputs_dict[step] = self.rc_inputs
            self.rc_outputs_dict[step] = self.rc_outputs
            self.rc_test_outputs_dict[step] = self.RC_model(self.rc_inputs, *optimized_params)

    def train_and_validate(self,t_df_1,v_df_1,leds,ac_T,ac_v,t_df_2,v_df_2,leds_2,ac_T_2,ac_v_2):#每个可能的时间格式弄一遍

        for step in self.steps_list:
            print(step)
            self.steps = step
            self.step_lenght = self.steps * 60
            optimized_params = self.train_one(t_df_1,v_df_1,leds,ac_T,ac_v)
            self.best_params_dict[step] = optimized_params
            self.rc_inputs_dict[step] = self.rc_inputs
            self.rc_outputs_dict[step] = self.rc_outputs
            self.rc_test_outputs_dict[step] = self.RC_model(self.rc_inputs, *optimized_params)
            self.optimized_params = optimized_params #临时存值
            self.validate_performance_dict[step] = self.validate(t_df_2,v_df_2,leds_2,ac_T_2,ac_v_2)
            self.validate_pred[step] = self.test_outputs
            self.validate_true[step] = self.rc_outputs
            
    def vel_train_and_validate(self,v_df_1,ac_v,v_df_2,ac_v_2):#每个可能的时间格式弄一遍
        for step in self.steps_list:
            self.steps = step
            self.step_lenght = self.steps * 60
            self.vel_train_one(v_df_1,ac_v)
            self.vel_validate_performance_dict[step] = self.vel_validate(v_df_2,ac_v_2)
            self.vel_black_box_models[step] = [self.linear_reg_model,self.svr_model,self.svm_model,self.lstm_model]
            
            
class spatial_RC():#空间分布rc管理类
    #存储每个节点的参数
    def __init__(self):
        self.rc_inputs_dict = {} #真实输入
        self.rc_outputs_dict = {} #真实输出
        self.rc_test_outputs_dict = {} #拟合输出
        
        self.best_mse_dict = {}
        self.best_params_dict = {}
        self.best_params_retrained_dict = {}
        self.validate_performance_dict = {}
        self.neighbor_dict = {}
        self.best_params_array = []
        self.validate_performance_array = []

        self.black_box_models = {}
        self.black_box_performances = {}
        
        self.validate_pred = {} #测试集预测输出
        self.validate_true = {} #测试集真实输出
        
        self.vel_neighbor_dict = {}
        self.vel_black_box_models = {}
        self.vel_validate_performance_dict = {}
        self.vel_validate_performance_array = []
        self.classes = classes

        self.previous_classes = {}  # Dictionary to store the previous class for each node
        self.class_order = list(nodes_dict.keys())  # Get the order of classes from the dictionary keys

        for k in ['a1', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15', 'a16', 'a17', 'a18', 'a19',
                  'a2', 'a20', 'a21', 'a22', 'a23', 'a24', 'a3', 'a4', 'a5', 'a6', 'a7'
                  , 'a8', 'a9', 'b1', 'b10', 'b11', 'b12', 'b13', 'b14', 'b15', 'b16', 'b17', 
                  'b18', 'b19', 'b2', 'b20', 'b21', 'b22', 'b23', 'b24', 'b3', 'b4', 'b5',
                  'b6', 'b7', 'b8', 'b9', 'c1', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15',
                  'c16', 'c17', 'c18', 'c19', 'c2', 'c20', 'c21', 'c22', 'c23', 'c24', 
                  'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'eject_node', 'return_node']:
            self.rc_inputs_dict[k] = {}
            self.rc_outputs_dict[k] = []
            self.rc_test_outputs_dict[k] = []
            self.best_mse_dict[k] = {}
            self.best_params_dict[k] = {}
            self.best_params_retrained_dict[k] = {}
            self.validate_performance_dict[k] = {}
            self.validate_pred[k] = []
            self.validate_true[k] = []
            self.neighbor_dict[k] = {}
            
        for i, current_class in enumerate(self.class_order):
            if i > 0 :#and current_class not in ['class2']:
                previous_class = self.class_order[i-1]  # Get the previous class based on the order
                for node in nodes_dict[current_class]:
                    self.previous_classes[node] = previous_class


    def spatial_params_train_and_validate(self,t_df_1,v_df_1,leds,ac_T,ac_v,t_df_2,v_df_2,leds_2,ac_T_2,ac_v_2):
        
        for node, previous_class in self.previous_classes.items():
            print(node)
            self.rc_model = multisteps_RC()
            self.rc_model.classes = self.classes
            self.rc_model.object = node
            self.rc_model.class_now = previous_class
            #print((self.rc_model.object,self.rc_model.class_now))
            self.rc_model.train_and_validate(t_df_1,v_df_1,leds,ac_T,ac_v,t_df_2,v_df_2,leds_2,ac_T_2,ac_v_2)
            #print(self.rc_model.validate_performance_dict)
            self.rc_inputs_dict[node] = self.rc_model.rc_inputs #dict of nodes of dict of different steps
            self.rc_outputs_dict[node] = self.rc_model.rc_outputs
            self.rc_test_outputs_dict[node] = self.rc_model.rc_test_outputs_dict
            self.best_params_dict[node] = self.rc_model.best_params_dict
            self.validate_performance_dict[node] = self.rc_model.validate_performance_dict
            
            self.best_params_array.append(self.rc_model.best_params_dict)
            self.validate_performance_array.append(self.rc_model.validate_performance_dict)
            self.neighbor_dict[node] = self.rc_model.neighbor_nodes
            self.validate_pred[node] = self.rc_model.validate_pred
            self.validate_true[node] = self.rc_model.validate_true
            
        return self.best_params_dict, self.validate_performance_dict
    
    def black_box_train_and_validate(self,t_df_1,v_df_1,leds,ac_T,ac_v,t_df_2,v_df_2,leds_2,ac_T_2,ac_v_2):
        for node, previous_class in self.previous_classes.items():
            self.rc_model = multisteps_RC()
            self.rc_model.object = node 
            self.rc_model.class_now = previous_class
            self.black_box_performances[node] = self.rc_model.data_treat_all(t_df_1,v_df_1,leds,ac_T,ac_v,t_df_2,v_df_2,leds_2,ac_T_2,ac_v_2)
            self.neighbor_dict[node] = self.rc_model.neighbor_nodes
            self.black_box_models[node] = {}
            self.black_box_models[node]['lr'] = self.rc_model.linear_reg_model
            self.black_box_models[node]['svr'] = self.rc_model.svr_model
            self.black_box_models[node]['svm'] = self.rc_model.svm_model
            self.black_box_models[node]['lstm'] = self.rc_model.lstm_model
            
    def device_only_black_box_train_and_validate(self,t_df_1,v_df_1,leds,ac_T,ac_v,t_df_2,v_df_2,leds_2,ac_T_2,ac_v_2):
        for node, previous_class in self.previous_classes.items():
            self.rc_model = multisteps_RC()
            self.rc_model.object = node 
            self.rc_model.class_now = previous_class
            self.black_box_performances[node] = self.rc_model.only_ac_data_treat_all(t_df_1,v_df_1,leds,ac_T,ac_v,t_df_2,v_df_2,leds_2,ac_T_2,ac_v_2)
            self.black_box_models[node] = {}
            self.black_box_models[node]['lr'] = self.rc_model.linear_reg_model
            self.black_box_models[node]['svr'] = self.rc_model.svr_model
            self.black_box_models[node]['svm'] = self.rc_model.svm_model
            self.black_box_models[node]['lstm'] = self.rc_model.lstm_model
    
    def stupid_black_box_train_and_validate(self,t_df_1,v_df_1,leds,ac_T,ac_v,t_df_2,v_df_2,leds_2,ac_T_2,ac_v_2):
        for node, previous_class in self.previous_classes.items():
            self.rc_model = multisteps_RC()
            self.rc_model.object = node 
            self.rc_model.class_now = previous_class
            self.black_box_performances[node] = self.rc_model.direct_data_treat_all(t_df_1,v_df_1,leds,ac_T,ac_v,t_df_2,v_df_2,leds_2,ac_T_2,ac_v_2)
            self.black_box_models[node] = {}
            self.black_box_models[node]['lr'] = self.rc_model.linear_reg_model
            self.black_box_models[node]['svr'] = self.rc_model.svr_model
            self.black_box_models[node]['svm'] = self.rc_model.svm_model
            self.black_box_models[node]['lstm'] = self.rc_model.lstm_model
            
                    
    def vel_spatial_black_box_train_and_validate(self,v_df_1,ac_v,v_df_2,ac_v_2):
        
        for node, previous_class in self.previous_classes.items():
            self.rc_model = multisteps_RC()
            self.rc_model.object = node
            self.rc_model.class_now = previous_class
            self.rc_model.vel_train_and_validate(v_df_1,ac_v,v_df_2,ac_v_2)
            
            self.vel_black_box_models[node] = self.rc_model.vel_black_box_models
            self.vel_validate_performance_dict[node] = self.rc_model.vel_validate_performance_dict
            
            self.vel_neighbor_dict[node] = self.rc_model.vel_neighbor_nodes
            self.validate_performance_array.append(self.rc_model.vel_validate_performance_dict)
            
        return self.validate_performance_array
            
            
            
            
            
            