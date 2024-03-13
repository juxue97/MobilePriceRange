import streamlit as st
import pickle
import tensorflow as tf
import numpy as np

def load_param():
    with open('saved_steps.pkl','rb') as file:
        data = pickle.load(file)
    loaded_model = tf.keras.models.load_model('my_model.keras')

    return data,loaded_model

data,model = load_param()

scaler = data['scaler']

def show_predict_page():
    #webpage
    st.title('Mobile Price Range Prediction Application')

    st.write('''##### Note:''')
    st.write('''Please fill your mobile details.''')

    def reset_session_state():
        pass

    n_cores_option = ('1','2','3','4','5','6','7','8')
    phone_g_option = ('No 3G','3G only','4G')

    with st.form("my_form"):
        col1,col2 = st.columns([1,1])
        with col1:
            blue = st.checkbox('blue',value=False,key='blue')
            dual_sim = st.checkbox('dual_sim',value=False)
            touch_screen = st.checkbox('touch_screen',value=False)
            wifi = st.checkbox('wifi',value=False)

        with col2:
            phone_g = st.radio('Type_g',phone_g_option)
            n_cores = st.selectbox('n_cores',n_cores_option)

        col1,col2 = st.columns([1,1])
        with col1:
            battery_power = st.text_input('Enter battery power:',value=0)
            clock_speed = st.text_input('Enter clock_speed:',value=0)
            int_memory = st.text_input('Enter int_memory:',value=0)
            m_dep = st.text_input('Enter m_dep:',value=0)
            mobile_wt = st.text_input('Enter mobile_wt:',value=0)
            ram = st.text_input('Enter ram:',value=0)

        with col2:
            fc = st.text_input('Enter fc:',value=0)
            pc = st.text_input('Enter pc:',value=0)

            px_h = st.text_input('Enter px_h:',value=0)
            px_w = st.text_input('Enter px_w:',value=0)

            sc_h = st.text_input('Enter sc_h:',value=0)
            sc_w = st.text_input('Enter sc_w:',value=0)

        col1,col2 = st.columns([0.15,1])
        with col1:
            ok = st.form_submit_button('Submit')
        with col2:
            reset = st.form_submit_button('Reset')

    def get_Bool(param):
        if param:
            return int(1)
        return int(0)

    def type_g(choice):
        #'No 3G','3G only','4G'
        if choice == 'No 3G':
            return 0
        elif choice == '3G only':
            return 1
        else:
            return 2

    if ok:
        st.write('LetsGO!! Start Predicting.')

        blue = int(get_Bool(blue))
        dual_sim = int(get_Bool(dual_sim))
        touch_screen = int(get_Bool(touch_screen))
        wifi = int(get_Bool(wifi))
        phone_g = int(type_g(phone_g))
        n_cores = int(n_cores)

        battery_power = float(battery_power)
        clock_speed = float(clock_speed)
        int_memory = float(int_memory)
        m_dep = float(m_dep)
        mobile_wt = float(mobile_wt)
        ram = float(ram)
        fcpc = float(fc) * float(pc)
        pxhw = float(px_h) * float(px_w)
        schw = float(sc_h) * float(sc_w)

        X = np.array([[blue,dual_sim,n_cores,touch_screen,wifi,phone_g,
                    battery_power,clock_speed,int_memory,m_dep,mobile_wt,ram,
                    fcpc,pxhw,schw]])
        #st.write(X[:,6:15])
        X[:,6:15] = scaler.transform(X[:,6:15])
        y_pred = np.argmax(model.predict(X),axis=1)
        st.subheader(f'The prediction price range= {y_pred}')
    '''
        ['blue', 'dual_sim', 'n_cores', 'touch_screen', 'wifi', 'phone_g',
        'battery_power', 'clock_speed', 'int_memory', 'm_dep', 'mobile_wt',
        'ram', 'fcpc', 'pxhw', 'schw']
    '''
    if reset:
        reset_session_state()