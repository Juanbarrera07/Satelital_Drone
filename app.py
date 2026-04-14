import streamlit as st
import os
import yaml
from pipeline import ingest, preprocess, classify, report

# Configuración de la página para un look corporativo Tier-1
st.set_page_config(page_title="TerraForge Mining Intelligence", layout="wide", page_icon="🛰️")

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    st.title("🛰️ TerraForge Mining Intelligence v2.0")
    st.subheader("Plataforma de Auditoría Geoespacial para Minería")
    
    config = load_config()
    
    # Sidebar de Control de Parámetros
    st.sidebar.header("⚙️ Configuración del Pipeline")
    # Adaptado para que coincida con las llaves que creamos en config.yaml
    proc_mode = config.get('pipeline', {}).get('processing_mode', 'fast')
    mode = st.sidebar.selectbox("Modo de Procesamiento", ["Fast", "Precision"], 
                               index=0 if proc_mode == 'fast' else 1)
    
    st.sidebar.divider()
    target_standard = config.get('reporting', {}).get('target_standard', 'IPCC_Tier_3')
    st.sidebar.info(f"Nivel de Cumplimiento: {target_standard}")

    # Dashboard Principal
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("### 1. Ingesta de Datos (ARD)")
        uploaded_file = st.file_uploader("Cargar Imagen Satelital (Sentinel-2 / Landsat)", type=["tif", "tiff"])
        
        if uploaded_file:
            st.success("Archivo cargado. Listo para validación BOA (Bottom of Atmosphere).")

    with col2:
        st.write("### 2. Análisis y Resultados")
        if st.button("Ejecutar Análisis Completo"):
            with st.spinner("Procesando con Windowed I/O y validando Quality Gates..."):
                # Aquí llamaremos a los módulos de la carpeta pipeline/
                st.info("Iniciando Segmentación de Fronteras Críticas (U-Net)...")
                st.success("Análisis Finalizado.")
                
                # Botón para descargar el reporte de credibilidad
                st.download_button(
                    label="📄 Descargar Informe de Credibilidad (PDF)",
                    data=b"Contenido del reporte", # Esto se conectará con pipeline/report.py
                    file_name="TerraForge_Audit_Report.pdf",
                    mime="application/pdf"
                )

if __name__ == "__main__":
    main()
