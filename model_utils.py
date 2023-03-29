from utils.plots import plot_one_box
from PIL import ImageColor
import subprocess
import streamlit as st
import psutil


def get_gpu_memory():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    return gpu_memory[0]


def get_system_stat(stframe1, stframe2, fps):
    # Updating Inference results
    with stframe1.container():
        st.markdown("<h2>Inference Statistics</h2>", unsafe_allow_html=True)
        if round(fps, 4)>1:
            st.markdown(f"<h4 style='color:green;'>Frame Rate: {round(fps, 4)}</h4>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h4 style='color:blue;'>Frame Rate: {round(fps, 4)}</h4>", unsafe_allow_html=True)
    
    with stframe2.container():
        st.markdown("<h2>System Statistics</h2>", unsafe_allow_html=True)
        js1, js2, js3 = st.columns(3)                       

        # Updating System stats
        with js1:
            st.markdown("<h4>Memory usage</h4>", unsafe_allow_html=True)
            mem_use = psutil.virtual_memory()[2]
            if mem_use > 50:
                js1_text = st.markdown(f"<h5 style='color:blue;'>{mem_use}%</h5>", unsafe_allow_html=True)
            else:
                js1_text = st.markdown(f"<h5 style='color:green;'>{mem_use}%</h5>", unsafe_allow_html=True)

        with js2:
            st.markdown("<h4>CPU Usage</h4>", unsafe_allow_html=True)
            cpu_use = psutil.cpu_percent()
            if mem_use > 50:
                js2_text = st.markdown(f"<h5 style='color:red;'>{cpu_use}%</h5>", unsafe_allow_html=True)
            else:
                js2_text = st.markdown(f"<h5 style='color:green;'>{cpu_use}%</h5>", unsafe_allow_html=True)

        with js3:
            st.markdown("<h4>GPU Memory Usage</h4>", unsafe_allow_html=True)  
            try:
                js3_text = st.markdown(f'<h5>{get_gpu_memory()} MB</h5>', unsafe_allow_html=True)
            except:
                js3_text = st.markdown('<h5>NA</h5>', unsafe_allow_html=True)
