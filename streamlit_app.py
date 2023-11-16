import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import pickle
from pysurvival.utils import load_model

st.set_page_config(layout="wide")

@st.cache_data(show_spinner=False)
def load_setting():
    settings = {
        'Age': {'values': [0, 21], 'type': 'slider', 'init_value': 10, 'add_after': ', year'},
        'Gender':{'values':["Female","Male"],'type': 'selectbox','init_value': 0,'add_after': ''},

        'Race': {'values': ["White", "Black","Asian or Pacific Islander","American Indian/Alaska Native", "Unknown"], 'type': 'selectbox', 'init_value': 0, 'add_after': ''}, 
        'Histological type': {'values': ["Astrocytic tumor", "Oligodendroglial tumor","Oligoastrocytic tumor","Ependymal tumor","Other gliomas","Unknown"],
                              'type': 'selectbox', 'init_value': 0, 'add_after': ''},  
        'Stage': {'values': ["I", "II","III","IV" ,"Unknown"],'type': 'selectbox', 'init_value': 0, 'add_after': ''},  
                             
        'Laterality': {'values': ["Left", "Right","Midline","Bilateral" ,"Unknown"],'type': 'selectbox', 'init_value': 0, 'add_after': ''}, 
        'Location of the tumor':{'values':["Supratentorial", "Infratentorial", "Midline", "Others"],'type':'selectbox','init_value':0,'add_after': ''},
        'Tumor size': {'values': [0, 200], 'type': 'slider', 'init_value': 38, 'add_after': ', mm'},
        'Tumor extension': {'values': ["Confined to Primary Location", "Ventricles", "Midline Crossing", "Extra-cerebral metastasis", "Unknown"],
                            'type': 'selectbox', 'init_value': 0, 'add_after': ''},
        'Surgery': {'values': ["Total gross resection", "Subtotal resection", "Biopsy", "No surgery", "Unknown"],'type': 'selectbox', 'init_value': 0, 'add_after': ''},
        'Radiation': {'values': ["Yes", "None/Unknown"],'type': 'selectbox', 'init_value': 0, 'add_after': ''},
        'Chemotherapy': {'values': ["Yes", "None/Unknown"],'type': 'selectbox', 'init_value': 0, 'add_after': ''},
        
    }
    input_keys = ['Age','Gender', 'Race','Histological type','Stage','Laterality', 'Location of the tumor',
                  'Tumor size',  'Tumor extension', 'Surgery', "Radiation", "Chemotherapy"]
    return settings, input_keys


settings, input_keys = load_setting()


@st.cache_data(show_spinner=False)
def get_model(name='DeepSurv'):
    # with open('./OutputModel/{}.pkl'.format(name), 'rb') as f:
    with open('./outputModel_1/deepsurv_model_1000.pkl', 'rb') as f:    
        model = pickle.load(f)
    return model


def get_code():
    sidebar_code = []
    
    for key in settings:
        if settings[key]['type'] == 'slider':
            sidebar_code.append(
                "{} = st.slider('{}',{},{},key='{}')".format(
                    key.replace(' ', '____'),
                    key + settings[key]['add_after'],
                    # settings[key]['values'][0],
                    ','.join(['{}'.format(value) for value in settings[key]['values']]),
                    settings[key]['init_value'],
                    key
                )
            )
        if settings[key]['type'] == 'selectbox':
            sidebar_code.append('{} = st.selectbox("{}",({}),{},key="{}")'.format(
                key.replace(' ', '____'),
                key + settings[key]['add_after'],
                ','.join('"{}"'.format(value) for value in settings[key]['values']),
                settings[key]['init_value'],
                key
            )
            )
    return sidebar_code




# print('\n'.join(sidebar_code))
if 'patients' not in st.session_state:
    st.session_state['patients'] = []
if 'display' not in st.session_state:
    st.session_state['display'] = 1
if 'model' not in st.session_state:
    st.session_state['model'] = 'DeepSurv'
deepsurv_model = get_model(st.session_state['model'])
sidebar_code = get_code()
def plot_survival():
    pd_data = pd.concat(
        [
            pd.DataFrame(
                {
                    'Survival': item['survival'],
                    'Time': item['times'],
                    'Patients': [item['No'] for i in item['times']]
                }
            ) for item in st.session_state['patients']
        ]
    )
    if st.session_state['display']:
        fig = px.line(pd_data, x="Time", y="Survival", color='Patients',range_x=[0,120], range_y=[0, 1])
    else:
        fig = px.line(pd_data.loc[pd_data['Patients'] == pd_data['Patients'].to_list()[-1], :], x="Time", y="Survival",
                      range_x=[0,120],range_y=[0, 1])
    fig.update_layout(template='simple_white',
                      title={
                          'text': 'Predicted Survival Probability',
                          'y': 0.95,
                          'x': 0.5,
                          'xanchor': 'center',
                          'yanchor': 'top',
                          'font': dict(
                              size=25
                          )
                      },
                      plot_bgcolor="white",
                      xaxis_title="Time (month)",
                      yaxis_title="Survival probability",
                      )
    st.plotly_chart(fig, use_container_width=True)


def plot_patients():
    patients = pd.concat(
        [
            pd.DataFrame(
                dict(
                    {
                        'Patients': [item['No']],
                        '1-Year': ["{:.2f}%".format(item['1-year'] * 100)],
                        '3-Year': ["{:.2f}%".format(item['3-year'] * 100)],
                        '5-Year': ["{:.2f}%".format(item['5-year'] * 100)]
                    },
                    **item['arg']
                )
            ) for item in st.session_state['patients']
        ]
    ).reset_index(drop=True)
    st.dataframe(patients)

# @st.cache_data(show_spinner=True)
def predict():
    print('update patients start . ##########')
    
    input = []
    
    for key in input_keys:
        value = st.session_state[key]
        if isinstance(value, int):
            input.append(value)
        if isinstance(value, str):
            input.append(settings[key]['values'].index(value))
    survival = deepsurv_model.predict_survival(np.array(input), t=None)
    # 保留120个预测的数据
    # survival = survival[:,0:120]
    # print(survival.shape)
    data = {
        'survival': survival.flatten(),
        'times': [i for i in range(0, len(survival.flatten()))],
        'No': len(st.session_state['patients']) + 1,
        'arg': {key:st.session_state[key] for key in input_keys},
        '1-year': survival[0, 12],
        '3-year': survival[0, 36],
        '5-year': survival[0, 60]
    }
    st.session_state['patients'].append(
        data
    )
    # print(len(survival.flatten()))
    print('update patients end ... ##########')

def plot_below_header():
    col1, col2 = st.columns([1, 9])
    col3, col4, col5, col6, col7 = st.columns([2, 2, 2, 2, 2])
    with col1:
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        # st.session_state['display'] = ['Single', 'Multiple'].index(
        #     st.radio("Display", ('Single', 'Multiple'), st.session_state['display']))
        st.session_state['display'] = ['Single', 'Multiple'].index(
            st.radio("Display", ('Single', 'Multiple'), st.session_state['display']))
        # st.radio("Model", ('DeepSurv', 'NMTLR','RSF','CoxPH'), 0,key='model',on_change=predict())
    with col2:
        plot_survival()
    with col4:
        st.metric(
            label='1-Year survival probability',
            value="{:.2f}%".format(st.session_state['patients'][-1]['1-year'] * 100)
        )
    with col5:
        st.metric(
            label='3-Year survival probability',
            value="{:.2f}%".format(st.session_state['patients'][-1]['3-year'] * 100)
        )
    with col6:
        st.metric(
            label='5-Year survival probability',
            value="{:.2f}%".format(st.session_state['patients'][-1]['5-year'] * 100)
        )
    st.write('')
    st.write('')
    st.write('')
    plot_patients()
    st.write('')
    st.write('')
    st.write('')
    st.write('')
    st.write('')

st.header('DeepSurv-based model for predicting survival of Pediatric Glioma', anchor='survival-of-Glioma')
if st.session_state['patients']:
    plot_below_header()
st.subheader("Instructions:")
st.write("1. Select patient's infomation on the left\n2. Press predict button\n3. The model will generate predictions")
st.write('***Note: this model is still a research subject, and the accuracy of the results cannot be guaranteed!***')
st.write("***[Paper link](https://www.baidu.com/)(Waiting for updates)***")
with st.sidebar:
    with st.form("my_form",clear_on_submit = False):
        for code in sidebar_code:
            exec(code)
        col8, col9, col10 = st.columns([3, 4, 3])
        with col9:
            prediction = st.form_submit_button(
                'Predict',
                on_click=predict,
                # args=[{key: eval(key.replace(' ', '____')) for key in input_keys}]
            )

