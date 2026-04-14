import streamlit as st

# Global UI safe-wrapper prevents application-level crashes from low level imports and bounds
try:
    from dotenv import load_dotenv
    load_dotenv()
    
    import leafmap.foliumap as leafmap
    import matplotlib.pyplot as plt
    import rasterio
    from rasterio.plot import show
    import numpy as np
    from pathlib import Path
    from datetime import date
    
    # Pipeline Core Native Imports
    from scripts.download_data import download_landsat, download_sentinel
    from pipeline.preprocess import Preprocessor
    from pipeline.analytics import SpectralAnalyzer

    st.set_page_config(
        page_title="TerraForge | Mining Intelligence Core",
        page_icon="⛏️",
        layout="wide"
    )

    # Standardize path routing purely via pathlib
    project_root = Path(__file__).resolve().parent

    st.sidebar.title("🛠️ TerraForge Control Panel")
    
    mission_platform = st.sidebar.selectbox(
        "🛰️ Mission Platform",
        ["Landsat 8/9 (USGS)", "Sentinel-2 (Copernicus)"]
    )
    
    # Global dynamic path isolation based on active sensor
    sensor_target = "sentinel" if "Sentinel" in mission_platform else "landsat"
    processed_dir = project_root / "data" / "processed" / sensor_target
    results_dir = project_root / "data" / "results" / sensor_target

    # Section routing configuration
    app_mode = st.sidebar.radio(
        "Navigation",
        ["1. Data Ingestion", "2. Tier-1 Analytics", "3. Tactical Visualization"]
    )

    # -----------------
    # MODULE 1: INGESTION
    # -----------------
    if app_mode == "1. Data Ingestion":
        sensor_target = "sentinel" if "Sentinel" in mission_platform else "landsat"
        archive_ext = "*.zip" if "Sentinel" in mission_platform else "*.tar"
        
        raw_dir = project_root / "data" / "raw" / sensor_target
        raw_dir.mkdir(parents=True, exist_ok=True)

        sensor_name = "USGS" if "Landsat" in mission_platform else "CDSE Copernicus"
        st.title(f"📡 M2M Data Ingestion ({sensor_name})")
        st.markdown(f"Automated M2M acquisition bridging remote satellite payloads directly into standard COGs.")
        
        with st.form("ingestion_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### AOI Configuration (Bounding Box)")
                min_lon = st.number_input("Min Longitude", value=-68.9, format="%.4f")
                max_lon = st.number_input("Max Longitude", value=-68.8, format="%.4f")
                min_lat = st.number_input("Min Latitude", value=-22.5, format="%.4f")
                max_lat = st.number_input("Max Latitude", value=-22.4, format="%.4f")
                
            with col2:
                st.markdown("##### Temporal & Cloud Criteria")
                start_d = st.date_input("Start Date", value=date(2025, 9, 1))
                end_d = st.date_input("End Date", value=date(2025, 11, 30))
                cloud_cover = st.slider("Max Cloud Cover (%)", 0, 100, 10)
                
            submit_ingest = st.form_submit_button(f"Download {sensor_name} Data")
            
        if submit_ingest:
            with st.spinner("Executing M2M bridge request. Keep session active (downloading GigaByte arrays)..."):
                st.toast(f"Authenticating {sensor_name} credentials...")
                
                s_date = start_d.strftime("%Y%m%d")
                e_date = end_d.strftime("%Y%m%d")
                
                before_files = set(raw_dir.glob(archive_ext))
                
                try:
                    if "Landsat" in mission_platform:
                        aoi_bbox = (min_lon, min_lat, max_lon, max_lat)
                        prog_bar = st.progress(0)
                        prog_text = st.empty()
                        
                        def download_cb(current, total, prefix="Processing"):
                            pct = current / total if total > 0 else 0
                            pct = min(max(pct, 0.0), 1.0)
                            prog_bar.progress(pct)
                            prog_text.text(f"{prefix} [{int(pct*100)}%]")
                            
                        download_landsat(aoi_bbox=aoi_bbox, start_date=s_date, end_date=e_date, max_cloud_cover=cloud_cover, progress_callback=download_cb)
                    else:
                        aoi_wkt = f"POLYGON(({min_lon} {min_lat}, {max_lon} {min_lat}, {max_lon} {max_lat}, {min_lon} {max_lat}, {min_lon} {min_lat}))"
                        st.info("Sentinel CDSE initialized. Review runtime environment logs for Sentinelsat telemetry.")
                        download_sentinel(aoi_wkt, s_date, e_date, cloud_cover=(0, cloud_cover))
                        
                    after_files = set(raw_dir.glob(archive_ext))
                    new_files = after_files - before_files
                    
                    if new_files:
                        st.success("Download complete!")
                        st.toast("Pushing raw archives to the Universal Refinery Pipeline...")
                        
                        for raw_archive in new_files:
                            with st.spinner(f"Refining COG archive natively: {raw_archive.name}"):
                                prep = Preprocessor()
                                prep.refinery_pipeline(str(raw_archive))
                                
                        st.success("✅ End-to-End Ingestion & Refinery Sequence Completed. Proceed to Analytics.")
                    else:
                        st.warning("Warning: No new files were resolved. The API returned empty results or payload was cached.")
                except Exception as e:
                    st.error(f"Critical Ingestion Error Mounted: {e}")

        # --- Mission Specific Local Intercept Blocks (Hybrid Flow) ---
        st.divider()
        raw_archives = list(raw_dir.glob(archive_ext))
            
        if raw_archives:
            st.markdown(f"##### Hybrid Flow: Process Existing {sensor_target.capitalize()} Archives")
            local_files = [f.name for f in raw_archives]
            
            # Secure Streamlit state by routing explicit unique keys based on target platform architecture
            ui_key_uid = sensor_target
            
            selected_files = st.multiselect(
                f"Select local {sensor_target} archives to process:", 
                local_files, 
                key=f"{ui_key_uid}_local_select"
            )
            
            if st.button("Refine Selected Archives to COG", key=f"{ui_key_uid}_refine_btn"):
                if not selected_files:
                    st.warning("⚠️ Please select at least one archive from the dropdown to continue processing.")
                else:
                    for filename in selected_files:
                        raw_archive = raw_dir / filename
                        st.toast(f"Processing local {sensor_target} resource: {filename}")
                        with st.spinner(f"Refining COG topology natively: {filename}"):
                            prep = Preprocessor()
                            prep.refinery_pipeline(str(raw_archive))
                    st.success(f"✅ {sensor_target.capitalize()} End-to-End Refinery Sequence Completed. Proceed to Analytics.")
                    st.stop()

    # -----------------
    # MODULE 2: ANALYTICS
    # -----------------
    elif app_mode == "2. Tier-1 Analytics":
        st.title("🔬 Tier-1 Spectral Analytics")
        st.markdown("Derives specialized geological signatures strictly validated against IPPC-compliant spatial rules.")
        
        available_sessions = [d.name for d in processed_dir.iterdir() if d.is_dir()] if processed_dir.exists() else []
        
        if not available_sessions:
            st.info("No processed sessions natively discovered. Navigate to Ingestion first.")
        else:
            selected_session = st.selectbox("Target Footprint Session", available_sessions)
            
            requested_products = st.multiselect(
                "Select Target Intelligence Products",
                ["TRUE_COLOR", "FALSE_COLOR_GEO", "NDVI", "NDWI", "CLAY", "IRON_OXIDE"],
                default=["NDVI", "TRUE_COLOR", "FALSE_COLOR_GEO"]
            )
            
            if st.button("Generate Intelligence"):
                with st.spinner("Processing massive arrays via RAM windowed chunk limits..."):
                    try:
                        prog_bar = st.progress(0)
                        prog_text = st.empty()
                        
                        def analytics_cb(current, total, prefix="Processing"):
                            pct = current / total if total > 0 else 0
                            pct = min(max(pct, 0.0), 1.0)
                            prog_bar.progress(pct)
                            prog_text.text(f"{prefix} [{int(pct*100)}%]")
                            
                        session_path = processed_dir / selected_session
                        analyzer = SpectralAnalyzer()
                        analyzer.generate_analytical_cogs(session_path, requested_products, progress_callback=analytics_cb)
                        
                        st.success(f"✅ Intelligence successfully assembled for footprint {selected_session}.")
                        st.toast("Resulting indices compiled to root 'data/results/'.")
                    except Exception as e:
                        st.error(f"Analytical block crash logged: {e}")

    # -----------------
    # MODULE 3: TACTICAL VISUALIZATION (VERSUS MODE)
    # -----------------
    elif app_mode == "3. Tactical Visualization":
        st.title("🗺️ Tactical Visualization: Versus Mode")
        st.markdown("Side-by-side prioritized comparison of geospatial sessions across multi-sensor environments. Expand plots fully natively using Streamlit toolbars.")
            
        def get_available_layers(session_path):
            if not session_path or not session_path.exists():
                return []
            layers = []
            sess_id = session_path.name
            for tif in session_path.glob("*_COG.tif"):
                layer_name = tif.name.replace("_COG.tif", "")
                if layer_name.startswith(f"{sess_id}_"):
                    layer_name = layer_name[len(sess_id)+1:]
                layers.append(layer_name)
            return sorted(set(layers))
            
        col_A, col_B = st.columns(2)
        
        with col_A:
            st.subheader("Left Engine")
            sensor_A = st.selectbox("Left Engine: Sensor", ["landsat", "sentinel"], key="sens_a")
            results_dir_A = project_root / "data" / "results" / sensor_A
            avail_A = [d.name for d in results_dir_A.iterdir() if d.is_dir()] if results_dir_A.exists() else []
            
            session_A = st.selectbox("Left Session", avail_A, key="sess_a") if avail_A else None
            avail_layers_A = get_available_layers(results_dir_A / session_A) if session_A else []
            default_A = ["TRUE_COLOR"] if "TRUE_COLOR" in avail_layers_A else (avail_layers_A[:1] if avail_layers_A else [])
            layers_A = st.multiselect("Left Layers", avail_layers_A, default=default_A, key="lay_a")
            
        with col_B:
            st.subheader("Right Engine")
            sensor_B = st.selectbox("Right Engine: Sensor", ["landsat", "sentinel"], key="sens_b")
            results_dir_B = project_root / "data" / "results" / sensor_B
            avail_B = [d.name for d in results_dir_B.iterdir() if d.is_dir()] if results_dir_B.exists() else []
            
            session_B = st.selectbox("Right Session", avail_B, key="sess_b") if avail_B else None
            avail_layers_B = get_available_layers(results_dir_B / session_B) if session_B else []
            default_B = ["TRUE_COLOR"] if "TRUE_COLOR" in avail_layers_B else (avail_layers_B[:1] if avail_layers_B else [])
            layers_B = st.multiselect("Right Layers", avail_layers_B, default=default_B, key="lay_b")
            
        st.divider()
        
        PRIORITY = ["TRUE_COLOR", "FALSE_COLOR_GEO", "NDVI", "NDWI", "CLAY", "IRON_OXIDE"]
        
        union_layers = list(set(layers_A + layers_B))
        union_layers.sort(key=lambda x: PRIORITY.index(x) if x in PRIORITY else 99)
        
        cmap_mapping = {
            "NDVI": "RdYlGn",
            "NDWI": "Blues",
            "CLAY": "YlOrBr",      
            "IRON_OXIDE": "Reds"   
        }
        
        def render_static_plot(tgt_results_dir, session_id, layer_name):
            if not session_id: return
            
            cog_path = tgt_results_dir / session_id / f"{session_id}_{layer_name}_COG.tif"
            if not cog_path.exists():
                st.error(f"Layer {layer_name} not natively found.")
                return
                
            try:
                with rasterio.open(cog_path.resolve().as_posix()) as src:
                    fig, ax = plt.subplots(figsize=(10, 10))
                    
                    if src.count == 3:
                        img_array = src.read()
                        img_array = np.transpose(img_array, (1, 2, 0)).astype(np.float32)
                        
                        valid_mask = img_array > 0
                        if np.any(valid_mask):
                            p2, p98 = np.percentile(img_array[valid_mask], (2, 98))
                        else:
                            p2, p98 = np.percentile(img_array, (2, 98))
                            
                        img_array = np.clip(img_array, p2, p98)
                        img_array = (img_array - p2) / (p98 - p2 + 1e-8)
                        
                        ax.imshow(img_array)
                        ax.set_title(f"{layer_name} Composition")
                    else:
                        cmap_choice = cmap_mapping.get(layer_name, "viridis")
                        show(src, ax=ax, cmap=cmap_choice, title=f"{layer_name} Signature")
                        
                    ax.set_axis_off()
                    st.pyplot(fig, clear_figure=True)
            except Exception as e:
                st.error(f"Render failure: {e}")
                
        for layer in union_layers:
            c1, c2 = st.columns(2)
            with c1:
                if layer in layers_A:
                    render_static_plot(results_dir_A, session_A, layer)
            with c2:
                if layer in layers_B:
                    render_static_plot(results_dir_B, session_B, layer)
                    
        # Dynamic Intelligence Guide mapping
        INTELLIGENCE_DICT = {
            "TRUE_COLOR": "**True Color (RGB):** Human-eye natural view. Provides baseline contextual awareness for infrastructure, active mining faces, and visible topography.",
            "FALSE_COLOR_GEO": "**False Color Geology (SWIR2-SWIR1-Red):** Cuts through atmospheric haze. Highly sensitive to lithological variations, rendering active mining zones and bare earth in sharp contrast.",
            "NDVI": "**NDVI (Vegetation):** Quantifies vegetation density. Critical for monitoring environmental compliance, forestry clearance, and rehabilitation of waste dumps.",
            "NDWI": "**NDWI (Moisture/Water):** Highlights surface water bodies and tailings moisture. Essential for detecting potential leakages in tailings storage facilities (TSF).",
            "CLAY": "**Clay Index (SWIR1/SWIR2):** Identifies hydrothermal alteration zones rich in clays. A key geotechnical indicator for assessing pit wall stability.",
            "IRON_OXIDE": "**Iron Oxide (Red/Blue):** Maps the spatial distribution of iron oxides. Used to detect rusting in waste dumps and delineate oxidized caps."
        }

        if layers_A or layers_B:
            with st.expander("📖 Tactical Intelligence Report", expanded=True):
                guide_col_A, guide_col_B = st.columns(2)
                
                with guide_col_A:
                    st.markdown(f"### Left Side: {sensor_A.capitalize()}")
                    if not layers_A:
                        st.info("No layers rendering.")
                    for lay in layers_A:
                        st.info(INTELLIGENCE_DICT.get(lay, f"**{lay}:** Custom Layer Representation."))
                        
                with guide_col_B:
                    st.markdown(f"### Right Side: {sensor_B.capitalize()}")
                    if not layers_B:
                        st.info("No layers rendering.")
                    for lay in layers_B:
                        st.info(INTELLIGENCE_DICT.get(lay, f"**{lay}:** Custom Layer Representation."))

except Exception as e:
    # Fail-safe root level UI catcher keeping the client inside the React frontend without a traceback crash loop
    st.error("🚨 Critical Subsystem Failure Intercepted.")
    st.error(f"Details: {e}")
    st.info("Please verify the conda/uv environments and ensure `localtileserver` is installed for Windows.")

