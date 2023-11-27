import streamlit as st
from PIL import Image
import pandas as pd 

def show_canopy():
    pictures_address = 'G:\Emist\Flow_surrogate\种植实验数据\图片重构'

    set = '二分之一'

    image = Image.open(f'{pictures_address}\{set}1.png')
    image2 = Image.open(f'{pictures_address}\{set}2.png')
    # Add a caption to the image
    caption = "The canopies of the city"

    # Display the image with a caption
    #st.image([image,image2], caption=caption)


    planting_densities = ["二分之一","三分之一","四分之一","六分之一","十二分之五"]

    unit_yield = [5.56, 7.75, 5.91, 6.83, 5.985] #g/head
    leaf_density = [30.86, 29.56, 21.44, 25.97, 20.36] #kg/m3
    light_flux = [48.67, 28.68, 23.14, 26.25, 27.72] #W/m2
    dark_flux = [-11.38, -4.83, -4.25, -3.96, -6.57] #W/m2
    min_wind_speed = [0.1, 0.25, 0.2, 0.2, 0.1] #m/s
    max_wind_speed = [0.2, 0.35, 0.5, 0.55, 0.25] #m/s

    impact_evaluation = [
        "风速分布均匀，整体较低，最高最低相差0.1m/s",
        "风速分布均匀，整体较低，风速最高和最低相差0.1m/s",
        "风速分布不均匀，整体较高，冠层风速最高和最低相差0.3m/s",
        "风速分布不均匀，整体较高冠层风速最高和最低相差0.35m/s",
        "风速分布均匀，整体较低，最高最低相差0.15m/s"
    ]


    @st.cache_data
    def get_dataset(number_of_items: int = 5, seed: int = 0) -> pd.DataFrame:
        new_data = []

        for i in range(number_of_items):

            new_data.append(
                {
                    "种植密度": planting_densities[i],
                    #"最终长势": f'{pictures_address}\{planting_densities[i]}1.png',
                    "单位产量(g/株)": unit_yield[i],
                    "平均密度(kg/m3)": leaf_density[i],
                    '光期热通量(W/m2)': light_flux[i],
                    '暗期热通量(W/m2)':dark_flux[i],
                    #'最小风速(m/s)':min_wind_speed[i],
                    #'最大风速(m/s)':max_wind_speed[i],
                    '影响评价':impact_evaluation[i]
                }
            )

        profile_df = pd.DataFrame(new_data)
        #profile_df["gender"] = profile_df["gender"].astype("category")
        return profile_df


    column_configuration = {
        "种植密度": st.column_config.TextColumn(
            "种植密度"
        ),
        #"最终长势": st.column_config.ImageColumn("最终长势", help="定植两周后的模样"),
        "单位产量(g/株)": st.column_config.NumberColumn("单位产量(g/株)"),
        "平均密度(kg/m3)": st.column_config.NumberColumn(
            "平均密度(kg/m3)", help="叶片的密度"
        ),
        "光期热通量(W/m2)": st.column_config.ProgressColumn(
            "光期热通量(W/m2)",min_value=0, max_value=50, format="%.2f"
        ),
        "暗期热通量(W/m2)": st.column_config.ProgressColumn(
            "暗期热通量(W/m2)",min_value=-20, max_value=0, format="%.2f"
        ),
        
        "影响评价": st.column_config.TextColumn(
            "影响评价",
            help="该种植情况下植物形态对流场影响的评价",
            #validate="^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)*$",
        ),
    }


    st.subheader('整体长势')
    st.data_editor(
        get_dataset(),
        column_config=column_configuration,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        #width=1200
    )
    
    st.subheader('三维结构捕捉过程')
    d_s = st.selectbox(label='选择种植密度', options=planting_densities)
    
    #col0, col1 = st.columns(2)
   #with col0:
    image = Image.open(f'{pictures_address}\{d_s}1.jpg')
    st.image(image,use_column_width=True)
    #with col1:
    #    image = Image.open(f'{pictures_address}\{d_s}2.png')
    #    st.image(image)
    col2,col3 = st.columns(2)
    
    with col2:
        image = Image.open(f'{pictures_address}\{d_s}2.png')
        st.image(image)
    with col3:
        image = Image.open(f'{pictures_address}\{d_s}4.png')
        st.image(image)
    
    image = Image.open(f'{pictures_address}\{d_s}风洞俯瞰.png')
    st.image(image)
    


    
