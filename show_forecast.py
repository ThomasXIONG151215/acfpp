import streamlit as st
from PIL import Image
import pandas as pd 
from graphviz import Digraph
from show_canopy import show_canopy
import plotly.express as px 

def show_forecast():
    st.subheader('核心思路')
    st.markdown("""
温度分布预测的方法专注于沿气流路径顺序，空间上滚动预测室内环境的气流和温度分布。讲究首先识别室内空气流动模式，明确观测节点，并将其根据气流经过的次序分入不同流场分段，确保模型准确刻画气流动态。然后，通过平滑数据处理和经验模态分解技术，精确捕捉温度变化趋势。在实际应用中，模型沿气流路径顺序，逐节点运用统计传热模型输出，实现对整个空间温度分布的连续预测。
""")
    st.markdown("""
最终实现，空间上的滚动预测，时间上的多步预测。
                """)
    
    st.subheader('可解释统计传热模型')
    # Part 1: Traditional RC Model
    st.markdown("""
    #### 传统 RC 模型
    在暖通空调领域，RC（电阻-电容）模型是预测建筑分区温度的常用模型，通过模拟热动力学行为预测和控制温度变化。其核心特点包括：
    1. **能量平衡原理**：模型基于一阶能量平衡原理，考虑建筑物内外的热交换和能量储存。
    2. **多区域控制适用性**：特别适合多区域建筑温度控制，能够处理不同区域间的热交换影响。
    """)
    # RC model equation
    st.markdown('''
        $$ 
        Q = C \\frac{dT}{dt} + \\sum R^{-1} (T_{\\text{indoor}} - T_{\\text{outdoor}}) 
        $$
    ''')
    
    # Part 2: Inputs for Polynomial Fitting Formulas of Thermal Resistances
    st.markdown("""
    #### 植物工厂场景下的热阻拟合公式输入
在植物工厂的应用中，对于植物工厂的特定应用，
研究人员在RC模型中完整保留了“C”（热容）部分。
但是，“R”（热阻）部分由于和气流状态密切相关，所以不能使用固定的热阻。
对流传热的热阻在不同空间位置是变化的，
采用了多项式公式来拟合以下热阻：
- **空调热阻**：输入空调风速。

- **邻居热阻**：输入邻居节点的平均风速。

- **LED灯热阻**：输入LED灯的散热量。

- **植物热阻**：输入植物的散热量。

    这些多项式公式能够根据输入变量拟合不同条件下的热阻，以精确计算节点的温度变化。
    """)
    
    st.markdown('''

    $$ 
    T_{\\text{node},t} = \\frac{1}{C} \\frac{\\left[ (T_{\\text{node},t-1} - T_{\\text{nb},t}) K_t \\right]}{R_{nb}} + Q_{\\text{ac}} R_{\\text{ac},t} + Q_{\\text{led}} R_{\\text{led},t} + Q_{\\text{plant}} R_{\\text{plant},t} 
    $$
''')
    st.subheader('应用效果')
    from PIL import Image
   
    

    data = pd.read_csv('3d_r2s.csv')
    image = Image.open(f'factory_streamlines.png')
    st.markdown('###### 新设计植物工厂气流特征')
    st.image(image=image,
            #caption=caption
            )

    st.markdown('###### CFD设定送风参数')
    image = Image.open(f'ac_curves_for_training.png')
    st.image(image)


    fig = px.scatter_3d(data, x='x', y='z', z='y', color='class')

    # Set the background color to white
    fig.update_layout(scene=dict(bgcolor='white'),
                    title='气流路径分段归纳'
                    )
    
    st.plotly_chart(fig)

    # Create a 3D scatter plot
    fig = px.scatter_3d(data, x='x', y='z', z='y', color='r2', color_continuous_scale='Blues')

    # Set the background color to white
    fig.update_layout(scene=dict(bgcolor='white'),
                    title='温度预测效果'
                    )

    st.plotly_chart(fig)
    
    data = {
    '分段': ['第一段', '第二段', '第三段', '第四段', '第五段', '第六段', '第七段'],
    '平均预测R2': [0.999, 0.912, 0.953, 0.935, 0.878, 0.902, 0.985],
    'color':[1,2,3,4,5,6,7]
    }

    df = pd.DataFrame(data)

    # Creating a bar plot using Plotly Express
    fig = px.bar(df, x='分段', y='平均预测R2', title='各分段的平均预测R2值',)

    st.plotly_chart(fig,theme='streamlit')
