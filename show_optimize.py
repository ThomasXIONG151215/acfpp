import pandas as pd 
import numpy as np
from scipy.optimize import minimize
import plotly.express as px
import streamlit as st
import json
from plotly.subplots import make_subplots

performance_table = {
    'comp_freq':[0,1,2,3,4,5,6,7],
    'power cost':[0,1.5,1.6,1.78,1.92,2.1,2,1.9],
    'cooling capacity':[0,4.3,5.3,6.2,6.9,8.0,7.4,7.2],
    'COP':[0,2.87,3.31,3.48,3.59,3.81,3.70,3.79]
}

fan_performance = {
    'fan_freq':[0,25,30,40,50],
}

fan_performance['power cost'] = []
fan_performance['ventilation'] = []

fan_k = 0.84/(50)**3

for freq in fan_performance['fan_freq']:
    fan_performance['power cost'].append(fan_k*freq**3)
    fan_performance['ventilation'].append(3900/3600*freq/50) #m3/s
    

fan_performance = pd.DataFrame(fan_performance)

performance_table = pd.DataFrame(performance_table)

def supply_calc(return_T,fresh_T, comp_freq, fan_freq, valve_state): #valve state: 0-1 新风比例
    cooling_capacity = performance_table[performance_table['comp_freq']==comp_freq]['cooling capacity'].values[0]
    COP = performance_table[performance_table['comp_freq']==comp_freq]['COP'].values[0]
    
    comp_cost = performance_table[performance_table['comp_freq']==comp_freq]['power cost'].values[0]
    fan_cost = fan_performance[fan_performance['fan_freq']==fan_freq]['power cost'].values[0]
    power_cost = comp_cost + fan_cost
    supply_v = fan_performance[fan_performance['fan_freq']==fan_freq]['ventilation'].values[0]
    rho = 1.2 #kg/m3
    cp = 1.005 #kJ/kg.K
    
    before_T = valve_state * fresh_T + (1-valve_state) * return_T
    
    if comp_freq != 0:
        supply_T = before_T - cooling_capacity/(COP*supply_v*rho*cp)
    else:
        supply_T = before_T
        
    return supply_T, supply_v, power_cost, cooling_capacity,COP
        
""" 
第一轮方案生成函数
"""

def dark_solution(nb_of_cycles,cool_to_heat):
    minimal_heat_capacity = 12 #kW, 上午10点至17点半夏季，16度相比较室外温差至少在9度以上，同时规定一旦开启就是最高水平风量
    # 定义目标函数，
    def objective(x):
        price_cost = 0
        for i in range(nb_of_cycles):
            time_now = x[i]+x[i+nb_of_cycles] + np.sum(x[:i]) + np.sum(x[nb_of_cycles:nb_of_cycles+i])
            #print(time_now)
            if time_now <= 5 * 60: #前五个小时是高峰时段 12-15
                price_cost += (x[i])/60 * 1.2 #分钟 * ￥/kWh-> ￥/kW
            elif time_now > 5 * 60 and time_now <= 7.5 * 60 + 10: #15-18平时段
                price_cost += (x[i])/60 * 0.7
                
        price_cost = price_cost * 0.84 #因为不需要制热，所以只需要开风机即可；内机送风时0.84 kW
     
        return price_cost

    # 定义约束条件，即每个周期的开关时间之和必须等于总开关时间，且每个周期的制冷量要能抵消总热量
    constraints = [{
        "type": "eq",
        "fun": lambda x: np.sum(x) - (7.5 * 60),
    }, {
        "type": "eq",
        "fun": lambda x: -np.sum(x[:nb_of_cycles]) * minimal_heat_capacity + cool_to_heat * 7.5 * 60, #总的开的时间现在是分钟，所以要抵消的热量也要转换为分钟
    }]
    
    bounds = [
        (5, 7.5 * 60) for i in range(2*nb_of_cycles)
    ]

    # 使用scipy.optimize.minimize函数找到最优解
    solution = minimize(objective, [0] * 2 * nb_of_cycles, constraints=constraints, method='SLSQP', 
                        bounds=bounds
                        )

    return solution

#光期方案生成器
def inside_solution(nb_of_cycles, heat_to_burn): #heat to burn是平时水平以kw统计
    # 定义目标函数，
    def objective(x):
        price_cost = 0
        for i in range(nb_of_cycles):
            time_now = x[i]+x[i+nb_of_cycles] + np.sum(x[:i]) + np.sum(x[nb_of_cycles:nb_of_cycles+i])
            #time_now = np.sum(x[nb_of_cycles:nb_of_cycles+i])
            if time_now <= 3 * 60: #前三个小时是高峰时段 18-21
                price_cost += (x[i])/60 * 1.2 #分钟 * ￥/kWh-> ￥/kW
            elif time_now > 3 * 60 and time_now <= 4 * 60:#21-22平时段
                price_cost += (x[i])/60* 0.7
            elif time_now > 4 * 60 and time_now < (4 + 8) * 60: #22-6 谷时段
                price_cost += (x[i])/60 * 0.3
            elif time_now >= (4 + 8) * 60 and time_now < 14 * 60: #6-8平时段
                price_cost += (x[i])/60* 0.7
            elif time_now >= 14 * 60: #8-11:30 高峰时段
                price_cost += (x[i])/60 * 1.2
                
        price_cost = price_cost * 2.94 #最后乘以最佳cop下的功耗，就有初始电费; 2.1是压缩机功率，送风机功率是0.84kW
        
        return price_cost

    # 定义约束条件，即每个周期的开关时间之和必须等于总开关时间，且每个周期的制冷量要能抵消总热量
    constraints = [{
        "type": "eq",
        "fun": lambda x: np.sum(x) - (15.5 * 60),
    }, {
        "type": "eq",
        "fun": lambda x: -np.sum(x[:nb_of_cycles]) * 8 + heat_to_burn * 15.5 * 60, #总的开的时间现在是分钟，所以要抵消的热量也要转换为分钟
    }]
    
    bounds = [
        (10, 15.5 * 60) for i in range(2*nb_of_cycles)
    ]

    # 使用scipy.optimize.minimize函数找到最优解
    solution = minimize(objective, [0] * 2 * nb_of_cycles, constraints=constraints, method='SLSQP', 
                        bounds=bounds
                        )

    return solution

def objective(x,nb_of_cycles):
    price_cost = 0
    for i in range(nb_of_cycles):
        time_now = x[i]+x[i+nb_of_cycles] + np.sum(x[:i]) + np.sum(x[nb_of_cycles:nb_of_cycles+i])
        print(time_now)
        
        if time_now <= 5 * 60: #前五个小时是高峰时段 12-15
            #print(x[i])
            print('高峰')
            price_cost += (x[i])/60 * 1.2 # h * ￥/kWh
            print(price_cost)
        elif time_now > 5 * 60 and time_now <= 7.5 * 60+3: #15-18平时段
            print('平')
            #print(x[i])
            price_cost += (x[i])/60 * 0.7
            print(price_cost)
            
    price_cost = price_cost * 0.84 #￥ #假设送风机自己开到最大是3kW
    
    return price_cost



def plot_sorted_solutions(dict):
    cycles = dict['cycles'][:8]
    on_times = dict['on_times'][:8]
    off_times = dict['off_times'][:8]
    price_costs = dict['price_costs'][:8]
    #st.write(on_times)
    return plot_solutions(cycles,on_times,off_times,price_costs)
    
def plot_solutions(cycles,on_times,off_times,price_costs):
    figs = []
    titles = []
    step = 1
    #可以再配个电价波动
    for solution in range(len(on_times)):
        #if cycles[solution] < 30:
        total_on_off_ys = []
        #st.write(on_times[solution][1])
        for periods_on,periods_off in zip(on_times[solution],off_times[solution]):
            for i in range(int(periods_on)):
                total_on_off_ys.append(1)
            for i in range(int(periods_off)):
                total_on_off_ys.append(0)
        
        fig = px.line(y=total_on_off_ys,
                    #height = 400
                    )
        
        fig.update_xaxes(title='时间/min')
        fig.update_yaxes(title='空调启停状态')
        title = '方案'+str(solution)+'-预计电费'+str(round(price_costs[solution],2))+'￥'
        fig.update_layout(title=title,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        #title_size=18
                        )
        titles.append(title)

        figs.append(fig)

    rows = int(len(figs)/2)
    fig = make_subplots(rows=rows,
                        cols=2,
                        #subplot_titles=titles
                        )
    
    for i in range(len(figs)):
        row = i // 2 + 1
        col = i % 2 + 1
        
        fig.add_trace(figs[i].data[0],
                    row=row,
                    col=col,
                    #subplot_titles=titles[i]
                    )
        fig.update_xaxes(#title_text="时间/min", 
                         title_standoff=0,
                            linecolor='black',  # X-axis border color
                            linewidth=2,        # X-axis border width
                            mirror=True,  # Mirror the line on all ticks
                            showgrid=False,     # Remove the grid lines
                            ticks='inside',    # Place ticks outside the plot
                            tickwidth=2,         # Ensure tick width is the same as line width
                            row=row,
                            col=col,
                            
                        )
        fig.update_yaxes(#title_text="空调启停状态", 
                         title_standoff=0,
                        linecolor='black',  # X-axis border color
                            linewidth=2,        # X-axis border width
                            mirror=True,  # Mirror the line on all ticks
                            showgrid=False,     # Remove the grid lines
                            ticks='inside',    # Place ticks outside the plot
                            tickwidth=2,         # Ensure tick width is the same as line width
                            row=row,
                            col=col,

                            )
    
        
    fig.update_layout(
        #width=500,
        height=200*rows,
        grid= {'rows': 2, 'columns': 2,
            'pattern': 'independent'},
        #font_size=10
    )
    fig.update_annotations(font=dict(size=14))
    fig.update_layout(
        #margin=dict(b=1000),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"  
    )
    fig.update_xaxes(matches='x')
    
    return fig

# Function to convert all NumPy arrays to lists in the dictionary
def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert array to list
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}  # Recurse into dict
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]  # Recurse into list
    else:
        return obj  # Return object as is if not numpy array or dict


#光期与暗期方案同时生成函数
def scheme_generator():
    cycles = [] #光期是从第一天晚上六点开始，然而算法是从头开始得所以cycles这个list的最后六个小时得调到前面
    dark_cycles = []
    price_costs = [] #第一阶段规划预计电费￥
    on_times = []
    off_times = []

    dark_price_costs = []
    dark_on_times = []
    dark_off_times = []

    light_solutions = {}
    dark_solutions = {}
    
    bar = st.progress(0,text='第一轮方案生成中...')
    
    for nb_of_cycles in range(2,50,1): #无论怎么样它都会直接平摊；那不如把电价波动也考虑进去；暂时如此；日程整体上可以保持那样
        bar.progress(nb_of_cycles*2,text='第一轮方案生成中...')
        test = inside_solution(nb_of_cycles,light_heat)
        dark_result = dark_solution(nb_of_cycles,dark_heat)
        
        if np.sum(test.x) >= 15.5*60-0.1 and np.sum(test.x) <= 15.5*60+0.1:
            cycles.append(nb_of_cycles)
            price_costs.append(test.fun)
            on_times.append(test.x[:nb_of_cycles])
            off_times.append(test.x[nb_of_cycles:])
        
        if np.sum(dark_result.x) >= 7.5*60 - 10 and np.sum(dark_result.x) <= 7.5*60 + 10 :
            dark_cycles.append(nb_of_cycles)
            dark_price_costs.append(dark_result.fun)
            dark_on_times.append(dark_result.x[:nb_of_cycles])
            dark_off_times.append(dark_result.x[nb_of_cycles:])

    bar.empty()
    st.success('第一轮方案生成完毕')
    #最后选择电费最低的前几个

    light_solutions['cycles'] = cycles
    light_solutions['price_costs'] = price_costs
    light_solutions['on_times'] = on_times
    light_solutions['off_times'] = off_times

    dark_solutions['cycles'] = dark_cycles
    dark_solutions['price_costs'] = dark_price_costs
    dark_solutions['on_times'] = dark_on_times
    dark_solutions['off_times'] = dark_off_times
    
    sorted_key = 'price_costs'
    sorted_dark_dict = {k: [v for _, v in sorted(zip(dark_solutions[sorted_key], dark_solutions[k]))] for k in dark_solutions}
    sorted_light_dict = {k: [v for _, v in sorted(zip(light_solutions[sorted_key], light_solutions[k]))] for k in light_solutions}

    sorted_dark_dict = convert_numpy(sorted_dark_dict)
    sorted_light_dict = convert_numpy(sorted_light_dict)
    
    with open('demo_1st_round_light_solutions.json','w') as file:
        json.dump(sorted_light_dict,file)
        
    with open('demo_1st_round_dark_solutions.json','w') as file:
        json.dump(sorted_dark_dict,file)
    
    light_fig = plot_sorted_solutions(sorted_light_dict)
    dark_fig = plot_sorted_solutions(sorted_dark_dict)
    
    return light_fig,dark_fig
    
""" 
流场代理模型训练
"""

import numpy as np
import random
start_time = 0

step = "1"

from multisteps_RC import multisteps_RC,spatial_RC,classes,t_df_1,t_df_2,v_df_1,v_df_2,ac_v,ac_v_2,leds,leds_2,ac_T,ac_T_2

rc_class = multisteps_RC()
rc_manage = spatial_RC()
rc_manage.rc_model = rc_class

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

rolling_t_df_1 = {k: rolling_average(t_df_1[k], 6) for k in t_df_1.keys()}
rolling_t_df_2 = {k: rolling_average(t_df_2[k], 6) for k in t_df_2.keys()}

rolling_t_df_1 = pd.DataFrame(rolling_t_df_1)-273.15
rolling_t_df_2 = pd.DataFrame(rolling_t_df_2)-273.15
v_df_1 = v_df_1[:-5]
leds = leds[:-5]
ac_T = ac_T[:-5]
ac_v = ac_v[:-5]
leds_2 = leds_2[:-5]
ac_T_2 = ac_T_2[:-5]
ac_v_2 = ac_v_2[:-5]

def prediction(comp_freq,fan_freq,valve_state, old_T_dict,return_T, fresh_T,led_state):#后面三个直接根据时间步来定
    
    ac_T,ac_v,power_cost, capacity, COP = supply_calc(return_T,fresh_T,comp_freq,fan_freq,valve_state)

    ctrc_new_T_dict = {}
    lr_new_T_dict = {}
    
    for node,previous_class in rc_manage.previous_classes.items():
        #print(node)
        if previous_class != 'class1':
            try:
                t_nb = np.mean([ctrc_new_T_dict[key] for key in rc_manage.neighbor_dict[node]])
            except:
                t_nb = np.mean([old_T_dict[key] for key in rc_manage.neighbor_dict[node]])
        else:
            t_nb = ac_T
        
        inputs = [
            led_state,
            ac_T,
            ac_v,
            old_T_dict[node],
            t_nb            
        ]
        #print(inputs)
        ctrc_new_T = rc_manage.rc_model.RC_model(inputs, *rc_manage.best_params_dict[node][step])
        ctrc_new_T_dict[node] = ctrc_new_T
        #print(ctrc_new_T)
        
        inputs = [inputs]
        lr_new_T = rc_manage.black_box_models[node]['lr'].predict(inputs)
        lr_new_T_dict[node] = lr_new_T
    
    
    #return inputs, lr_new_T_dict,ctrc_new_T_dict
    return ctrc_new_T_dict 

normal_nodes = []
absurd_nodes = []
for key in t_df_2.keys():
    val = np.mean(t_df_2[key])-273.15
    if val > 34 or key in ['TimeStep']:
    #if key in ['TimeStep','a1']:
        absurd_nodes.append(key)
    else:
        normal_nodes.append(key)

rolling_t_df_1 = rolling_t_df_1[normal_nodes]
rolling_t_df_2 = rolling_t_df_2[normal_nodes]
v_df_1 = v_df_1[normal_nodes]
v_df_2 = v_df_2[normal_nodes]

from emd import EMD_total

sum_IMF_dict = {}
residual_dict = {}
sum_IMF_dict_2 = {}
residual_dict_2 = {}
for k in rolling_t_df_1.keys():
    if k not in ['TimeStep','flow-time',
                 'a1',
    'a11',
    'a22',
    'a8',
    'b10',
    'b18',
    'b19',
    'b2',
    'b21',
    'b24',
    'c11'
    ]:
        print(k)
        sum_IMF_dict[k],residual_dict[k] = EMD_total(k,rolling_t_df_1)
        sum_IMF_dict_2[k],residual_dict_2[k] = EMD_total(k,rolling_t_df_2)
    else:
        sum_IMF_dict[k] = np.array(rolling_t_df_1[k])/2
        residual_dict[k] = np.array(rolling_t_df_1[k])/2
        sum_IMF_dict_2[k] = np.array(rolling_t_df_2[k])/2
        residual_dict_2[k] = np.array(rolling_t_df_2[k])/2

def train_model():
    imf_rc_manage = spatial_RC()
    res_rc_manage = spatial_RC()
    #imf_rc_manage.classes = {k: imf_rc_manage.classes[k] for k in normal_nodes[1:]}
    #res_rc_manage.classes = {k: res_rc_manage.classes[k] for k in normal_nodes[1:]}

    new_classes = {}
    for clas in imf_rc_manage.classes.keys():
        new_classes[clas] = []
        for key in imf_rc_manage.classes[clas]:
            if key in normal_nodes:
                new_classes[clas].append(key)
        
    imf_rc_manage.classes = new_classes
    res_rc_manage.classes = new_classes


    imf_rc_manage.previous_classes = {key: imf_rc_manage.previous_classes[key] for key in normal_nodes[1:]}
    res_rc_manage.previous_classes = {key: res_rc_manage.previous_classes[key] for key in normal_nodes[1:]}
    imf_rc_manage.neighbor_dict = {key: imf_rc_manage.neighbor_dict[key] for key in normal_nodes[1:]}
    res_rc_manage.neighbor_dict = {key: res_rc_manage.neighbor_dict[key] for key in normal_nodes[1:]}

    bar = st.progress(0,text='流场代理模型训练中...')
    imf_rc_manage.spatial_params_train_and_validate(sum_IMF_dict,v_df_1,leds,ac_T,ac_v,
                            sum_IMF_dict_2,v_df_2,leds_2,ac_T_2,ac_v_2)
    bar.progress(50,text='流场代理模型训练中...')
    res_rc_manage.spatial_params_train_and_validate(residual_dict,v_df_1,leds,ac_T,ac_v,
                            residual_dict_2,v_df_2,leds_2,ac_T_2,ac_v_2)
    bar.progress(100,text='流场代理模型训练中...')
    bar.empty()
    st.success('流场代理模型训练完毕，可以用于温度分布预测')
    return imf_rc_manage, res_rc_manage

""" 
第二轮方案调整函数
"""

class flow_predictive_control():
    def __init__(self,imf_rc_manage,res_rc_manage,t_df,imf,res,weathers,leds):
        self.imf_rc_manage = imf_rc_manage
        self.res_rc_manage = res_rc_manage
        
        self.ref_t_df = t_df
        self.t_recording = {}
        self.imf_recording = {}
        self.res_recording = {}
                
        self.weather_forecast = weathers
        self.leds = leds
        
        self.count_runs = 0
        
        for node in normal_nodes:
            self.t_recording[node] = []
            self.imf_recording[node] = []
            self.res_recording[node] = []
            
            self.t_recording[node].append(pd.DataFrame(t_df).iloc[-1,:]['a1']-0.7)#假设大家初始情况下都一样 #取样本数据集的最后一行作为起始值
            self.imf_recording[node].append(pd.DataFrame(imf).iloc[-1,:]['a1']-0.7)
            self.res_recording[node].append(pd.DataFrame(res).iloc[-1,:]['a1']-0.7)
        #print(self.t_recording['return_node'])
        self.comp_freqs = []
        self.fan_freqs = []
        self.valve_states = []
        self.ac_Ts = []
        self.ac_vs = []
        self.power_costs = []
        self.capacities = []
        self.COPs = []
            
        self.time = 0 #t
        self.obj_T = 22
        
        self.classes = classes
    
    def copy_object(self,t_df,imf,res,weathers,leds):
        self.copy = flow_predictive_control(t_df,imf,res,weathers,leds)
        self.copy.classes = classes 
        self.copy.t_recording = self.t_recording
        self.copy.imf_recording = self.imf_recording
        self.copy.res_recording = self.res_recording
        #self.copy.new_T_dict = self.new
        
        return self.copy
    
    def t_history_update(self):
        #print('update')
        for node in normal_nodes:
            self.t_recording[node].append(self.new_T_dict[node])
            self.imf_recording[node].append(self.new_imf_dict[node])
            self.res_recording[node].append(self.new_res_dict[node])
    
    def t_reinitiate(self):
        for node in normal_nodes:
            self.t_recording[node][-1] = 22 + random.uniform(-0.5,0.5)
            self.imf_recording[node][-1] = 11 + random.uniform(-0.5,0.5)
            self.res_recording[node][-1] = 11 + random.uniform(-0.5,0.5)
    
    def t_prediction(self, comp_freq, fan_freq,valve_state):
        self.count_runs += 1
        self.led_state = self.leds[-1]
        self.fresh_T = self.weather_forecast[-1][1]
        self.return_T = self.t_recording['return_node'][-1]
        #print(self.fresh_T)
        #print(self.return_T)
        #print((self.return_T,self.fresh_T,comp_freq,fan_freq,valve_state))
        self.ac_T,self.ac_v,self.power_cost, self.capacity, self.COP = supply_calc(self.return_T,self.fresh_T,comp_freq,fan_freq,valve_state)
        
        old_t_dict = {node:self.t_recording[node][-1] for node in self.t_recording.keys()}
        old_imf_dict = {node:self.imf_recording[node][-1] for node in self.imf_recording.keys()}
        old_res_dict = {node:self.res_recording[node][-1] for node in self.res_recording.keys()}
        
        self.new_T_dict = {} #临时
        self.new_imf_dict = {}
        self.new_res_dict = {}

        for node in normal_nodes:
            #print(node)
            if node == 'a1':
                self.new_T_dict[node] = self.ac_T
                self.new_imf_dict[node] = self.ac_T/2
                self.new_res_dict[node] = self.ac_T/2
            else:
                try:
                    imf_nb = np.mean([self.new_imf_dict[key] for key in self.imf_rc_manage.neighbor_dict[node]])
                    res_nb = np.mean([self.new_res_dict[key] for key in self.res_rc_manage.neighbor_dict[node]])
                    t_nb = np.mean([self.new_T_dict[key] for key in rc_manage.neighbor_dict[node]])
                except:
                    imf_nb = np.mean([old_imf_dict[key] for key in self.imf_rc_manage.neighbor_dict[node]])
                    res_nb = np.mean([old_res_dict[key] for key in self.res_rc_manage.neighbor_dict[node]])
                    t_nb = np.mean([old_t_dict[key] for key in self.imf_rc_manage.neighbor_dict[node]])

                imf_inputs = [
                    self.led_state,
                    self.ac_T,
                    self.ac_v,
                    old_imf_dict[node],
                    imf_nb            
                ]
                
                res_inputs = [
                    self.led_state,
                    self.ac_T,
                    self.ac_v,
                    old_res_dict[node],
                    res_nb   
                ]
                
                t_inputs = [
                    self.led_state,
                    self.ac_T,
                    self.ac_v,
                    old_t_dict[node],
                    t_nb
                ]
                #print(imf_inputs)
                #print(res_inputs)
                self.new_imf_dict[node] = self.imf_rc_manage.rc_model.RC_model(imf_inputs, *self.imf_rc_manage.best_params_dict[node][step])
                self.new_res_dict[node] = self.res_rc_manage.rc_model.RC_model(res_inputs, *self.res_rc_manage.best_params_dict[node][step])
            
            #print(t_inputs)
            #print(np.array(t_inputs).reshape(1,-1))
                #self.new_T_dict[node] = rc_manage.black_box_models[node]['lr'].predict(np.array(t_inputs).reshape(1,-1))[0]
                self.new_T_dict[node] = self.new_imf_dict[node] + self.new_res_dict[node]
            #print(self.new_T_dict[node])
            #print(old_t_dict[node])
            import random
            if abs(self.new_T_dict[node] - old_t_dict[node]) > 4: #预测的有问题的话，就依照参考数据集的差值平均来重新算一遍一个随机值
                ref_diff = 2#abs(np.mean(self.ref_t_df.diff()[node]))
                self.new_T_dict[node] = old_t_dict[node] + random.uniform(-ref_diff,ref_diff)

            self.imf_recording[node].append(self.new_imf_dict[node])
            self.res_recording[node].append(self.new_res_dict[node])
            self.t_recording[node].append(self.new_T_dict[node])
            
        return self.new_T_dict 

    def t_distribution(self,comp_freq, fan_freq,valve_state):
        ctrc = self.t_prediction(comp_freq,fan_freq,valve_state)
        self.classes_characteristics = {}
        #打分制
        #平均温度与22度相差0.8以上
        points = 0
        for i in range(1,6):
            self.classes_characteristics['class ' + str(i+1)] = {}
            self.classes_characteristics['class ' + str(i+1)]['all'] = [ctrc[k] for k in normal_nodes[1:] if rc_manage.previous_classes[k] == 'class' + str(i+1)]
            self.classes_characteristics['class ' + str(i+1)]['mean'] = np.mean(self.classes_characteristics['class ' + str(i+1)]['all'])
            self.classes_characteristics['class ' + str(i+1)]['var'] = np.var(self.classes_characteristics['class ' + str(i+1)]['all'])
            if abs(self.obj_T -self.classes_characteristics['class ' + str(i+1)]['mean']) > 1:
                #print('class'+str(i))
                #print((self.obj_T,self.classes_characteristics['class ' + str(i+1)]['mean']))
                points += 1
            if self.classes_characteristics['class ' + str(i+1)]['var'] > 3:
                #print('class'+str(i))
                #print(self.classes_characteristics['class ' + str(i+1)]['var'])
                points += 0.5
            #print(points)
            #print('class'+str(i))
            #print(self.classes_characteristics['class ' + str(i+1)]['mean'])
            #print(self.classes_characteristics['class ' + str(i+1)]['var'])
        return points

def time_range(on_times,off_times):#提取具体什么时候开空调什么时候不开空调
    accumulate_time = 0
    on_pivots = []
    off_pivots = []
    for ont,offt in zip(on_times,off_times):
        on_pivots.append((accumulate_time,accumulate_time+ont))
        accumulate_time += ont 
        off_pivots.append((accumulate_time,accumulate_time+offt))
        accumulate_time += offt
    return on_pivots,off_pivots

def forced_optimization(before_change_points,before_change_power_cost,flow_controller):
    after_change_points = before_change_points
    best_comp_freq = 7
    best_fan_freq = 50
    best_valve_state = 0.5
    best_power_cost = 0
    for comp_freq in performance_table['comp_freq']:
        for fan_freq in range(30,50,10):
            for valve_state in range(0,1):
                temporal_controller = copy.deepcopy(flow_controller) #每次要更新下
                print(len(temporal_controller.t_recording['a3']))
                change_points = temporal_controller.t_distribution(comp_freq,fan_freq,valve_state) #self.t得有所更新
                #print((change_points,after_change_points))
                if change_points < after_change_points:
                    #print((comp_freq,fan_freq,valve_state))
                    #print(change_points)
                    after_change_points = change_points
                    best_comp_freq = comp_freq
                    best_fan_freq = fan_freq
                    best_valve_state = valve_state
                    best_power_cost = temporal_controller.power_cost
                elif change_points == after_change_points and temporal_controller.power_cost < before_change_power_cost:
                    #print((comp_freq,fan_freq,valve_state))
                    #print(change_points)
                    after_change_points = change_points
                    best_comp_freq = comp_freq
                    best_fan_freq = fan_freq
                    best_valve_state = valve_state
                    best_power_cost = temporal_controller.power_cost

                
                
    return after_change_points, best_power_cost, (best_comp_freq, best_fan_freq, best_valve_state)

import copy

def control_conditioning(initial_controls,flow_controller):
    temporal_controller = copy.deepcopy(flow_controller)
    before_change_point = temporal_controller.t_distribution(*initial_controls)
    before_change_t_dict = temporal_controller.classes_characteristics
    print(('before_change_point',before_change_point))
    print(temporal_controller.classes_characteristics)
    before_change_power_cost = temporal_controller.power_cost
    if before_change_point > 2:
        print('need change')
        #运行
        control_record = 'need change'
        after_change_point, final_power_cost,final_controls = forced_optimization(before_change_point,before_change_power_cost,flow_controller)
        print((after_change_point,final_controls))
    else:
        after_change_point = before_change_point
        final_controls = initial_controls
        final_power_cost = before_change_power_cost
        control_record = 'no need change'
    
    #然后才根据时间记录历史
    new_T = flow_controller.t_distribution(*final_controls) #跑一次模型然后在flow controller本身内部里头记录
    #flow_controller.t_history_update()
    after_change_t_dict = flow_controller.classes_characteristics

    return before_change_point,after_change_point,control_record, final_controls, before_change_t_dict, after_change_t_dict,before_change_power_cost,final_power_cost

light_leds = [1 for i in range(int(16*60/5))] #光期led状态，待检查暗期情况

def operation_predictive_planning(imf,res,initial_schedules_package,period_length_hours,nb_sols):

    solutions = {}
    for j in range(nb_sols):
        solutions[j] = {}
        on_pivots, off_pivots = time_range(initial_schedules_package['on_times'][j],initial_schedules_package['off_times'][j])

        flow_controller = flow_predictive_control(imf,res,rolling_t_df_2,sum_IMF_dict_2,residual_dict_2,weather_predict,light_leds)
        flow_controller.obj_T = 22
        
        before_change_points = [] #惩罚因子
        after_change_points = []
        before_change_controls = []
        after_change_controls = []
        control_records = []

        before_change_t_dict = []
        after_change_t_dict = []

        count_meet_on_pivots = 0
        count_meet_off_pivots = 0

        before_power_costs = []
        after_power_costs = []
        
        final_controls = []


        for i in range(int(period_length_hours*60/5)):
            time_in_min = i * 5
            flow_controller.time += 5
            print(('time in min',time_in_min)) 
            if i%4 == 0:
                flow_controller.t_reinitiate()
            for ont,offt in zip(on_pivots,off_pivots):
                if time_in_min >= int(ont[0]) and time_in_min < int(ont[1]):
                    print('ac on') #运行正常空调优化
                    count_meet_on_pivots += 1
                    
                    initial_controls = (7,50,0.5)     
                    
                    before_change_controls.append(initial_controls)
                    #print('before change')
                    before_point, after_point, control_record, final_control, before_state, after_state, before_power_cost, after_power_cost = control_conditioning(initial_controls,flow_controller)
                    #print('after change')
                    before_change_points.append(before_point)
                    after_change_points.append(after_point)
                    control_records.append(control_record)
                    after_change_controls.append(final_control)   
                    before_change_t_dict.append(before_state)
                    after_change_t_dict.append(after_state)
                    before_power_costs.append(before_power_cost)
                    after_power_costs.append(after_power_cost)
                    
                elif time_in_min >= int(offt[0]) and time_in_min < int(offt[1]):
                    #print((time_in_min,int(offt[0]),int(offt[1])))
                    print('ac off')
                    count_meet_off_pivots += 1
                    
                    initial_controls = (0,0,0.5)#完全不开情况>
                    
                    before_change_controls.append(initial_controls)
                    
                    before_point, after_point, control_record, final_control, before_state, after_state, before_power_cost, after_power_cost = control_conditioning(initial_controls,flow_controller)
                    #print('after change')
                    before_change_points.append(before_point)
                    after_change_points.append(after_point)
                    control_records.append(control_record)
                    after_change_controls.append(final_control)   
                    before_change_t_dict.append(before_state)
                    after_change_t_dict.append(after_state)
                    before_power_costs.append(before_power_cost)
                    after_power_costs.append(after_power_cost)
            
        solutions[j]['before_change_points'] = before_change_points
        solutions[j]['after_change_points'] = after_change_points
        solutions[j]['before_change_t_dict'] = before_change_t_dict
        solutions[j]['after_change_t_dict'] = after_change_t_dict
        solutions[j]['before_change_controls'] = before_change_controls
        solutions[j]['before_change_comp_freq'] = [control[0] for control in before_change_controls]
        solutions[j]['before_change_fan_freq'] = [control[1] for control in before_change_controls]
        solutions[j]['before_change_valve_state'] = [control[2] for control in before_change_controls]
        
        solutions[j]['after_change_controls'] = after_change_controls
        solutions[j]['after_change_comp_freq'] = [control[0] for control in after_change_controls]
        solutions[j]['after_change_fan_freq'] = [control[1] for control in after_change_controls]
        solutions[j]['after_change_valve_state'] = [control[2] for control in after_change_controls]
        
        solutions[j]['before_power_costs'] = before_power_costs
        solutions[j]['after_power_costs'] = after_power_costs
        
    return solutions
def second_round_sort(solution):
    new_t_dis = {k: np.mean(v['after_change_points']) for k,v in solution.items()}
    sorted_keys = sorted(new_t_dis,key=new_t_dis.get)
    sorted_solution = dict()
    for k in sorted_keys:
        sorted_solution[k] = solution[k]
    return sorted_solution
def second_round_sort(solution):
    new_t_dis = {k: np.mean(v['after_change_points']) for k,v in solution.items()}
    sorted_keys = sorted(new_t_dis,key=new_t_dis.get)
    sorted_solution = dict()
    for k in sorted_keys:
        sorted_solution[k] = solution[k]
    return sorted_solution

def all_second_round_subplots(stage,solution,nb_sols):
    #nb_sols多少个方案要展示
    param_befs = ['before_power_costs','before_change_comp_freq','before_change_fan_freq','before_change_valve_state','before_change_points']
    param_afs = ['after_power_costs','after_change_comp_freq','after_change_fan_freq','after_change_valve_state','after_change_points']
    param_realnames = ['能耗(kW)','外机挡位','内机频率(Hz)','新风比例','温度分布扣分']
    color_befs = []
    color_afs = []
    st.markdown('### '+stage)
    for name,bef,af in zip(param_realnames,param_befs,param_afs):
        st.write(name)
        fig = one_group_of_subplots(stage,solution,nb_sols,name,bef,af)
        st.plotly_chart(fig)
        st.markdown('***')
def one_group_of_subplots(stage,solution,nb_sols,param_realname,param_bef,param_af):
    figs = []
    for i in range(nb_sols):#
        key = list(solution.keys())[i]
        #print(len(solution[key][param_bef]))
        #print(len(solution[key][param_af]))
        fig = px.line(solution[key],y=[param_bef,param_af])
        #fig.update_layout(legend=dict(x=0.05, y=1))
        #names = []
        #for i,stuff in enumerate(rolling_t_df_2.iloc[:-2,:][[param_bef,param_af]]):
        #    names.append(i)
        #fig.data[names[0]].name=f"{node}节点温度"
        #fig.data[names[1]].name=f"HTS模型预测{node}节点温度"
        
        figs.append(fig)
    rows = int(len(figs)/2)
    fig = make_subplots(rows=rows,
                        cols=2,
                        #subplot_titles=titles
                        )
    
    for i in range(len(figs)):
        key = list(solution.keys())[i]
        row = i//2 + 1
        col = i%2 + 1
        print((row,col))
        trace_bef = figs[i].data[0]
        trace_af = figs[i].data[1]

        # Customize the legend names and colors
        trace_bef = trace_bef.update(name=f'方案{key}-初步规划', marker=dict(color='lightblue'))
        trace_af = trace_af.update(name=f'方案{key}-调整后', marker=dict(color='darkred'))

        fig.add_trace(trace_bef,
                      row=row,
                      col=col)
        
        fig.add_trace(trace_af,
                      row=row,
                      col=col)
    
    for i in range(1, rows + 1):
        for j in range(1, 3):
            fig.update_xaxes(title_text="时间/min", linecolor='black', linewidth=2, mirror=True,
                            showgrid=False, ticks='inside', tickwidth=2, row=i, col=j)
            fig.update_yaxes(title_text=param_realname, linecolor='black', linewidth=2, mirror=True,
                            showgrid=False, ticks='inside', tickwidth=2, title_standoff=0, row=i, col=j)
                            # The title_standoff is set to 5 here, adjust as needed to bring the title closer
            fig.update_layout(title=f'{stage}方案第二轮调整{param_realname}前后对比')
    """
    fig.update_xaxes(title_text="时间/min", 
                                linecolor='black',  # X-axis border color
                                linewidth=2,        # X-axis border width
                                mirror=True,  # Mirror the line on all ticks
                                showgrid=False,     # Remove the grid lines
                                ticks='inside',    # Place ticks outside the plot
                                tickwidth=2,         # Ensure tick width is the same as line width
                                row=row,
                                col=col,
                            )
        fig.update_yaxes(title_text=param_realname, 
                        linecolor='black',  # X-axis border color
                            linewidth=2,        # X-axis border width
                            mirror=True,  # Mirror the line on all ticks
                            showgrid=False,     # Remove the grid lines
                            ticks='inside',    # Place ticks outside the plot
                            tickwidth=2,         # Ensure tick width is the same as line width
                            row=row,
                            col=col,

                            )
    """
        
    fig.update_layout(
        width=1100,
        height=300*rows,
        grid= {'rows': 2, 'columns': 2,
            'pattern': 'independent'},
        font_size=16
    )
    fig.update_annotations(font=dict(size=24))
    fig.update_layout(
        #margin=dict(b=1000),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"  
    )
    fig.update_xaxes(matches='x')
    
    return fig

""" 
第三轮统计分析
"""
def recount(solution,nb_sols):
    third_round = pd.DataFrame({})


    bp = []
    ap = []

    bpt = []
    apt = []
    bfa = []
    afa = []

    cr = []

    for i in range(nb_sols):
        i = str(i)
        #print(i)
        #print('power costs change')
        #print(np.sum(light_optimal_controls[i]['before_power_costs'])/(12))
        #print(np.sum(light_optimal_controls[i]['after_power_costs'])/(12))
        bp.append(np.sum(solution[i]['before_power_costs'])/(12))
        ap.append(np.sum(solution[i]['after_power_costs'])/(12))
        
        #print('temperature distribution points')
        #print(np.mean(light_optimal_controls[i]['before_change_points']))
        #print(np.mean(light_optimal_controls[i]['after_change_points']))
        bpt.append(np.mean(solution[i]['before_change_points']))
        apt.append(np.mean(solution[i]['after_change_points']))
        bfa.append(np.mean(solution[i]['before_change_valve_state']))
        afa.append(np.mean(solution[i]['after_change_valve_state']))

        ventilations = [fan_performance[fan_performance['fan_freq']==freq]['ventilation'].values[0] * ratio 
                        for freq,ratio in zip(solution[i]['after_change_fan_freq'],
                                              solution[i]['after_change_valve_state'])]
        total_v = np.mean(ventilations)
        
        average_change_rate = total_v * 3600 /33.5 #m3/h /m3
        cr.append(average_change_rate)
    
    third_round['换算平均每小时换气次数'] = cr
    third_round['第一轮总能耗'] = bp
    third_round['第二轮总能耗'] = ap
    third_round['第一轮平均温度扣分'] = bpt
    third_round['第二轮平均温度扣分'] = apt
    third_round['第一轮平均新风比例'] = bfa
    third_round['第二轮平均新风比例'] = afa
    
    #light_second_round['cycles'] = solution['cycles'][:nb_sols]
    third_round['方案'] = list(solution.keys())[:nb_sols]
    
    third_round = third_round.set_index('方案')
    
    return third_round

""" 
一起上
"""

single_unit_led_power = 16*2
total_number_of_units = 128
total_power = single_unit_led_power * total_number_of_units

led_heat = total_power*0.7 + total_power * 0.3 * 0.15 #kW
single_unit_plant_heat_dark = -18 / 1000 #kW
signle_unit_plant_heat_light = 28 / 1000

plant_heat_dark = single_unit_plant_heat_dark * total_number_of_units #kW 
plant_heat_light = signle_unit_plant_heat_light * total_number_of_units
plant_heat_light = 2
#light_heat = led_heat
#led_heat
light_heat = 6.53#kW
dark_heat = 0.57#kW

weather_predict = [(0, 23.604024966097118),
  (3600, 24.778765210050125),
  (7200, 23.185347795123842),
  (10800, 24.036928067973246),
  (14400, 24.569258573931787),
  (18000, 25.63509091508683),
  (21600, 26.78796944937326),
  (25200, 27.83615800796737),
  (28800, 28.088335495818527),
  (32400, 29.264718122661794),
  (36000, 29.06095419768741),
  (39600, 32.080714964895904),
  (43200, 33.01398338364424),
  (46800, 33.835193643273904),
  (50400, 30.427229473913894),
  (54000, 31.9820835275271),
  (57600, 27.94648322938749),
  (61200, 29.244366996795964),
  (64800, 27.962247933963067),
  (68400, 28.529559691206916),
  (72000, 28.78329071694235),
  (75600, 26.473396341705154),
  (79200, 25.147314187098356),
  (82800, 25.676265131815896)]

from PIL import Image
import time 
def show_optimize():
    #bar = st.progress(0)

    #with st.spinner(text='In progress'):  
    #    time.sleep(5)
    #    st.success('Done')
    #    bar.progress(100)
        
    st.subheader('空调流场预测规划')
    
    st.markdown('空调流场预测规划(ACFPP)同时优化运行电费，能耗与温度分布')
    # Data to be displayed in the dataframe
    logic = {
        "规划轮次": ["第一轮", "第二轮", "第三轮"],
        "最优规划目标": ["预计电费", "温度分布以及达标率", "每小时换气次数"],
        "解决问题": [
            "如何编配日程可使运行成本最低？",
            "如何保障温度分布均匀？",
            "如何保证空气质量？"
        ],
        "涉及参数": [
            "周期数，满负荷启停空调时长",
            "空调外机与内机具体挡位和频率参数",
            "新风比例"
        ]
    }

    # Create a DataFrame
    df = pd.DataFrame(logic)

    # Display the dataframe in Streamlit
    st.data_editor(df)
    
    
    case_tab, first_tab,sur_tab,second_tab,third_tab = st.tabs(['工况设定',
                                                                'ACFPP第一轮方案生成',
                                                                '训练流场代理模型',
                                                                'ACFPP第二轮方案优化',
                                                                'ACFPP第三轮方案优选'])
    
    

    
    with case_tab:
        st.markdown('### 工况设定')
        st.markdown('###### 上海夏季典型气温')
        
        weather = pd.read_csv('acfpp_weather.csv')
        fig = px.line(weather,x='time',y='outdoor_T')
        st.plotly_chart(fig)
        
        st.markdown('###### 电价波动与周期变化')
        image = Image.open(f'elec_price_and_period.png')
        st.image(image)
    
    with first_tab:
        st.markdown('### ACFPP第一轮生成光期空调运行方案')
        st.markdown('用尽量最少的电费让热量抵消')
        round_1_container = st.container()
        
        col1,col2 = st.columns(2)
        with col1:
            recalculate = st.button('重新计算')
        with col2:
            load = st.button('加载方案')
        
        if recalculate:
            light_fig,dark_fig = scheme_generator()
        
        if load:
            with open('demo_1st_round_light_solutions.json','r') as file:
                sorted_light_dict = json.load(file)
                
            with open('demo_1st_round_dark_solutions.json','r') as file:
                sorted_dark_dict = json.load(file)
            
            light_fig = plot_sorted_solutions(sorted_light_dict)
            dark_fig = plot_sorted_solutions(sorted_dark_dict)
        with round_1_container:
            if recalculate or load:
                st.markdown('###### 光期空调运行方案')
                st.plotly_chart(light_fig,use_container_width=True,theme='streamlit')
                st.markdown('###### 暗期空调运行方案')
                st.plotly_chart(dark_fig,use_container_width=True,theme='streamlit')

    with sur_tab:
        st.markdown('#### 流场代理模型训练')
        st.markdown('快速预测温度分布')
        col1,col2 = st.columns(2)
        with col1:
            recalculate = st.button('重新计算',key='surrogate model')
        with col2:
            load = st.button('加载方案',key='surrogate model load')
        if recalculate:
            imf_rc_manage, res_rc_manage = train_model()
            imf_rc_manage.neighbor_dict = convert_numpy(imf_rc_manage.neighbor_dict)
            res_rc_manage.neighbor_dict = convert_numpy(res_rc_manage.neighbor_dict)
            imf_rc_manage.best_params_dict = convert_numpy(imf_rc_manage.best_params_dict)
            res_rc_manage.best_params_dict = convert_numpy(res_rc_manage.best_params_dict)
            
            #存储相邻节点信息
            with open('demo_imf_rc_neighbor_dict.json','w') as file:
                json.dump(imf_rc_manage.neighbor_dict,file)
            with open('demo_res_rc_neighbor_dict.json','w') as file:
                json.dump(res_rc_manage.neighbor_dict,file)
            
            with open('demo_imf_rc_params_dict.json','w') as file:
                json.dump(imf_rc_manage.best_params_dict,file)
            with open('demo_res_rc_params_dict.json','w') as file:
                json.dump(res_rc_manage.best_params_dict,file)
        if load:
            #空白的预测器和训练后的预测器之间的区别在于邻近节点信息和模型参数
            imf_rc_manage = spatial_RC()
            res_rc_manage = spatial_RC()
            with open('demo_imf_rc_neighbor_dict.json','r') as file:
                imf_rc_manage.neighbor_dict = json.load(file)
            with open('demo_res_rc_neighbor_dict.json','r') as file:
                res_rc_manage.neighbor_dict = json.load(file)
            
            with open('demo_imf_rc_params_dict.json','r') as file:
                imf_rc_manage.best_params_dict = json.load(file)
            with open('demo_res_rc_params_dict.json','r') as file:
                res_rc_manage.best_params_dict = json.load(file)
            st.success('加载完毕')
    with second_tab:
        st.markdown('### ACFPP第二轮调整优化')
        st.markdown('优化温度分布于能耗')
        
        with open('demo_1st_round_light_solutions.json','r') as file:
            sorted_light_dict = json.load(file)
            
        with open('demo_1st_round_dark_solutions.json','r') as file:
            sorted_dark_dict = json.load(file)
        
        imf_rc_manage = spatial_RC()
        res_rc_manage = spatial_RC()
        imf_rc_manage.rc_model = multisteps_RC()
        res_rc_manage.rc_model = multisteps_RC()
        with open('demo_imf_rc_neighbor_dict.json','r') as file:
            imf_rc_manage.neighbor_dict = json.load(file)
        with open('demo_res_rc_neighbor_dict.json','r') as file:
            res_rc_manage.neighbor_dict = json.load(file)
        
        with open('demo_imf_rc_params_dict.json','r') as file:
            imf_rc_manage.best_params_dict = json.load(file)
        with open('demo_res_rc_params_dict.json','r') as file:
            res_rc_manage.best_params_dict = json.load(file)
            
        st.success('第一轮方案于流场代理模型均已加载完毕')
        
        col1,col2 = st.columns(2)
        with col1:
            recalculate = st.button('重新计算',key='secound round')
        with col2:
            load = st.button('加载方案',key='secound round load')
        
        if recalculate:
            #try:
            light_optimal_controls = operation_predictive_planning(imf_rc_manage, res_rc_manage,sorted_light_dict,15.5,8)
            dark_optimal_controls = operation_predictive_planning(imf_rc_manage, res_rc_manage,sorted_dark_dict,7.5,4)
            
            sorted_light_solutions = second_round_sort(light_optimal_controls)
            sorted_dark_solutions = second_round_sort(dark_optimal_controls)
            
            with open('demo_2nd_round_light_solutions.json','w') as file:
                json.dump(sorted_light_solutions,file)
            with open('demo_2nd_round_dark_solutions.json','w') as file:
                json.dump(sorted_dark_solutions,file)
            
            #except:
             #   st.error('请先训练流场代理模型')
        
        if load:
            try:
                with open('demo_2nd_round_light_solutions.json','r') as file:
                    sorted_light_solutions = json.load(file)
                
                with open('demo_2nd_round_dark_solutions.json','r') as file:
                    sorted_dark_solutions = json.load(file)
               
            except:
                st.error('请先训练流场代理模型')
        
        if recalculate or load:
            #try:
            #直接画图
            all_second_round_subplots('光期',sorted_light_solutions,4)
            all_second_round_subplots('暗期',sorted_dark_solutions,2)
            #st.plotly_chart(second_lights)
            #st.plotly_chart(second_darks)
            #except:
                #pass
    with third_tab:
        st.markdown('### ACFPP第三轮优选')
        st.markdown('将各个方案的新风比例与送风量结合换算出平均每小时换气次数')
        
        with open('demo_2nd_round_light_solutions.json','r') as file:
            sorted_light_solutions = json.load(file)
        
        with open('demo_2nd_round_dark_solutions.json','r') as file:
            sorted_dark_solutions = json.load(file)
        
        load = st.button('加载',key='third round load')
        #try
        light_recount = recount(sorted_light_solutions,8)
        dark_recount = recount(sorted_dark_solutions,4)
        st.markdown('#### 光期方案')
        st.data_editor(light_recount)
        st.markdown('#### 暗期方案')
        st.data_editor(dark_recount)
        #except:
        #    st.error('请先完成第二轮迭代')
        
        
