import streamlit as st 
from PIL import Image
import pandas as pd

butterfly_refs = {
"Reference (Chicago Format)": [
    'Kastner, P., & Dogan, T. (2021). "Eddy3D: A toolkit for decoupled outdoor thermal comfort simulations in urban areas." Building and Environment.',
    'Young, E., et al. (2021). "Modeling outdoor thermal comfort along cycling routes at varying levels of physical accuracy to predict bike ridership in Cambridge, MA." Building and Environment.',
    'De Simone, Z., et al. (2021). "Towards Safer Work Environments During the COVID-19 Crisis: A Study of Different Floor Plan Layouts and Ventilation Strategies." Building Simulation 2021 Conference.',
    'Dogan, T., et al. (2021). "Surfer: A fast simulation algorithm to predict surface temperatures and mean radiant temperatures in large urban models." Journal of Building and Environment.',
    'Dogan, T., et al. (2020). "Streamlined CFD simulation framework for wind-pressure coefficients on building facades." Journal of Building Simulation.',
    'Natanian, J., et al. (2020). "From energy performative to livable Mediterranean cities: An annual outdoor thermal comfort and energy balance cross-climatic typological study." Energy & Buildings.',
    'Kastner, P., et al. (2020). "Predicting space usage by multi-objective assessment of outdoor thermal comfort around a university campus." SimAUD 2020.',
    'Kastner, P., et al. (2020). "Solving Thermal Bridging Problems for Architectural Applications with OpenFOAM." SimAUD 2020.'
],
"Publication Date & Conference/Journal": [
    '2021, Building and Environment',
    '2021, Building and Environment',
    '2021, Building Simulation Conference',
    '2021, Journal of Building and Environment',
    '2020, Journal of Building Simulation',
    '2020, Energy & Buildings',
    '2020, SimAUD',
    '2020, SimAUD'
],
"Brief Description": [
    'Developed a toolkit for outdoor thermal comfort simulations in urban areas.',
    'Modeled outdoor thermal comfort along cycling routes in Cambridge, MA.',
    'Examined floor plan layouts and ventilation strategies for safer work environments during COVID-19.',
    'Developed a simulation algorithm for surface temperatures in urban models.',
    'Created a CFD simulation framework for wind-pressure coefficients on buildings.',
    'Studied annual outdoor thermal comfort and energy balance in Mediterranean cities.',
    'Assessed outdoor thermal comfort to predict space usage around a university campus.',
    'Addressed thermal bridging problems for architectural applications with OpenFOAM.'
]
}

honeybee_refs = {
"Reference (Chicago Format)": [
    'Ganji, H. B., Utzinger, D. M., Bradley, D. E. "Create and Validate Hybrid Ventilation Components in Simulation using Grasshopper and Python in Rhinoceros.” Sep 2019, 16th IBPSA International Conference, Rome',
    'Maffessanti, Viola. "Wind and Urban Spaces. Evaluation of a CFD Parametric Framework for Early-Stage Design." Jun 2019, 4th IBPSA-Italy Conference, Bozen-Bolzano',
    'Mackey, Christopher, et al. “Wind, Sun, Surface Temperature, and Heat Island: The Critical Variables for High‐Resolution Outdoor Thermal Comfort.” Aug 2017, 15th International Conference of Building Performance Simulation Association, San Francisco',
    'Salamone, Francesco, et al. "Design and Development of a Nearable Wireless System to Control Indoor Air Quality and Indoor Lighting Quality." 2017, Sensors Journal',
    'Fang, Yuan. “Optimization of Daylighting and Energy Performance Using Parametric Design, Simulation Modeling, and Genetic Algorithms.” Spring 2017, North Carolina State University Dissertation',
    'Konis, Kyle, et al. “Passive performance and building form: An optimization framework for early-stage design support.” Feb 2016, Solar Energy Journal'
],
"Publication Date & Conference/Journal": [
    'Sep 2019, 16th IBPSA International Conference, Rome',
    'Jun 2019, 4th IBPSA-Italy Conference, Bozen-Bolzano',
    'Aug 2017, 15th International Conference of Building Performance Simulation Association, San Francisco',
    '2017, Sensors Journal',
    'Spring 2017, North Carolina State University Dissertation',
    'Feb 2016, Solar Energy Journal'
],
"Brief Description": [
    'Developed and validated hybrid ventilation components in simulations.',
    'Evaluated a CFD parametric framework for urban spaces and wind analysis in early design stages.',
    'Focused on critical variables affecting outdoor thermal comfort, including wind and sun.',
    'Developed a wireless system to control indoor air and lighting quality.',
    'Optimized daylighting and energy performance through parametric design and simulation.',
    'Provided an optimization framework for passive performance and building form in early design.'
]
}

honeybee_refs = pd.DataFrame(honeybee_refs)
butterfly_refs = pd.DataFrame(butterfly_refs)


from graphviz import Digraph

def create_flowchart():
    # Create a directed graph
    dot = Digraph(comment='强排方法过程', format='svg')

    # Add nodes
    dot.node('A', '1. 引入强排技术')
    dot.node('B', '2. 参数化生成算例')
    dot.node('C', '3. 得到最优的5个方案')
    dot.node('D', '4. 寻找市面上匹配产品')
    dot.node('E', '5. 匹配情况判断')
    dot.node('F', '回到第2步')
    dot.node('G', '6. 设计运营')

    # Add edges
    dot.edges(['AB', 'BC', 'CD'])
    dot.edge('D', 'E', label='检查匹配')
    dot.edge('E', 'F', label='匹配不到')
    dot.edge('E', 'G', label='匹配到')

    # Loop back
    dot.edge('F', 'B')

    return dot



def show_plan():
    st.markdown('# PFAL设计技术路线探讨')
    
    # Displaying the design focus in Markdown format
    st.markdown('''
    ### 集装箱植物工厂设计重点

    1. **产量**：种植架摆放和作物之间的关系。
    2. **能耗**：进行温湿度仿真，进行产品设备选型。
    3. **产量与能耗耦合**：处理植物散热与蒸腾问题。
    ''')

    # Displaying the first step of design in Markdown format
    st.markdown('''
    ---

    ### 第一步：结构设计
    - **技术方法**：
        1. 引入强排技术。
        2. 参数化生成算例。
        3. 在产量和能源方面各自得到最优的5个方案。
        4. 寻找市面上匹配产品。
        5. 如果匹配不到，回到第2步调整；如果匹配到，继续到第6步。
        6. 设计运营。

    - **应用范围**：
        1. **种植架强排**：考虑可能种植的作物形态来确定最佳摆放方式。现成产品较少，但定制难度小。
        2. **空调通风设计**：
            - 能耗-Honeybee插件，结合EnergyPlus与Rhino，批量计算能量平衡。
            - 流场-Butterfly插件，结合OpenFOAM与Rhino，批量计算多相流场情况。
        3. **反光膜+LED灯**：基于CWY的积累和Honeybee的光照模拟功能，在设计阶段先节省50%。
    ''')
    
    #st.graphviz_chart(create_flowchart())
    
    
    image = Image.open(f'后续软件关系图.png')

    # Add a caption to the image
    caption = "rhino/grasshopper/ladybugtools软件生态"

    # Display the image with a caption
    st.image(image, caption=caption)
    
    # Displaying the second step of design in Markdown format
    st.markdown('''
    ---

    ### 第二步：实现运行控制
    - 将现有工作标准化->批量生成更多集装箱设计方案的空调运行方案。
    - 从Fluent生成的CFD数据来源变更为rhino/gh/ladybugtools。
    ''')

    # Displaying the third step of design in Markdown format
    st.markdown('''
    ---

    ### 第三步：材料与合作
    - 探索与BKL和YZT合作，研究围护结构材料的可能性。
    ''')
    
    st.markdown('''
    ---

    ### 工具可靠性综述
    ''')
    
    st.markdown('#### Honeybee')
    st.data_editor(honeybee_refs)
    
    st.markdown('#### Butterfly')
    st.data_editor(butterfly_refs)

    st.markdown('## PFAL系统选型平台')
    st.markdown('### 运营指标')
    col1,col2 = st.columns(2)
    with col1:
        st.selectbox('选择地区',options=['北京','上海','广州','迪拜','莫斯科','纽约','东京','新德里','墨尔本','圣保罗'])
        st.selectbox('选择时节',options=['夏季','冬季','春季','秋季'])
    
    with col2:
        st.select_slider('选择作物',options=['小白菜','生菜','莴苣','芥兰','菠菜','茼蒿','香菜','芹菜','韭菜','苋菜','韭黄','油菜','芥菜','甘蓝','芜菁','菜心','苦菊','莴笋','莴苣','芦笋','蒿子杆','茴香','芥蓝','芥菜','莴苣','芹菜','韭菜','苋菜','韭黄','油菜','芥菜','甘蓝','芜菁','菜心','苦菊','莴笋','莴苣','芦笋','蒿子杆','茴香','芥蓝','芥菜','莴苣','芹菜','韭菜','苋菜','韭黄','油菜','芥菜','甘蓝','芜菁','菜心','苦菊','莴笋','莴苣','芦笋','蒿子杆','茴香','芥蓝','芥菜','莴苣','芹菜','韭菜','苋菜','韭黄','油菜','芥菜','甘蓝','芜菁','菜心','苦菊','莴笋','莴苣','芦笋','蒿子杆','茴香','芥蓝','芥菜'])
        st.select_slider('目标产量(kg)',options=['100','200','300','400','500','600','700','800','900','1000'])
    
    calc = st.button('种植架强排计算')
    if calc:
        st.success('计算完成，适合的集装箱尺寸为xxxxx')
    
    col3,col4 = st.columns(2)
    with col3:
        st.markdown('### 围护结构结构')
        
        st.slider('导热系数',min_value=0.1,max_value=1.0,step=0.1)
        st.slider('反光率',min_value=0.1,max_value=1.0,step=0.1)
        st.slider('透光率',min_value=0.1,max_value=1.0,step=0.1)
    
    with col4:
        
        st.markdown('### 设备性能')
        st.selectbox('空调选型',options=['空调1','空调2','空调3','空调4','空调5','空调6','空调7','空调8','空调9','空调10'])
        st.selectbox('辐射制冷',options=['是','否'])
        st.selectbox('LED灯选型',options=['LED1','LED2','LED3','LED4','LED5','LED6','LED7','LED8','LED9','LED10'])
        
    calc = st.button('空调能源优化计算')
    if calc:
        st.metric('预计每日电费','0.1元')
        st.metric('预计能耗','0.1kWh')
        st.metric('产能比','5kWh/kg')
    
    
