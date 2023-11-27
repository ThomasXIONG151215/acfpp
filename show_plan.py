import streamlit as st 


def show_plan():
    st.markdown('## PFAL系统选型平台Demo')
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
    