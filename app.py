import streamlit as st

# Global UI safe-wrapper prevents application-level crashes from low level imports and bounds
try:
    from dotenv import load_dotenv
    load_dotenv()
    
    import leafmap.foliumap as leafmap
    from pathlib import Path
    from datetime import date
    
    # Pipeline Core Native Imports
    from scripts.download_data import download_landsat
    from pipeline.preprocess import Preprocessor
    from pipeline.analytics import SpectralAnalyzer

    st.set_page_config(
        page_title="TerraForge | Mining Intelligence Core",
        page_icon="⛏️",
        layout="wide"
    )

    # Standardize path routing purely via pathlib
    project_root = Path(__file__).resolve().parent
    processed_dir = project_root / "data" / "processed"
    results_dir = project_root / "data" / "results"
    raw_dir = project_root / "data" / "raw" / "landsat"

    st.sidebar.title("🛠️ TerraForge Control Panel")

    # Section routing configuration
    app_mode = st.sidebar.radio(
        "Navigation",
        ["1. Data Ingestion (USGS)", "2. Tier-1 Analytics", "3. Tactical Visualization"]
    )

    # Global Intelligence Guide contextualized gracefully into sidebar
    with st.sidebar.expander("📖 Index Intelligence Guide"):
        st.markdown("""
        * **NDVI (Normalized Difference Vegetation Index):** Tracks vegetation health and rehabilitation progress.
        * **NDWI (Normalized Difference Water Index):** Identifies water bodies, tailings moisture, and potential leakages.
        * **Clay Index:** Highlights hydrothermal alteration zones rich in clay minerals (SWIR1/SWIR2). Crucial for pit wall stability and lithology mapping.
        * **Iron Oxide:** Detects oxidation in waste rock dumps and mineral deposits (Red/Blue). High values indicate rusting or hematite presence.
        """)

    # -----------------
    # MODULE 1: INGESTION
    # -----------------
    if app_mode == "1. Data Ingestion (USGS)":
        st.title("📡 M2M Data Ingestion (USGS)")
        st.markdown("Automated M2M acquisition bridging remote satellite payloads directly into standard COGs.")
        
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
                
            submit_ingest = st.form_submit_button("Download USGS Data")
            
        if submit_ingest:
            with st.spinner("Executing M2M bridge request. Keep session active (downloading GigaByte arrays)..."):
                st.toast("Authenticating USGS credentials...")
                
                s_date = start_d.strftime("%Y%m%d")
                e_date = end_d.strftime("%Y%m%d")
                aoi_bbox = (min_lon, min_lat, max_lon, max_lat)
                
                raw_dir.mkdir(parents=True, exist_ok=True)
                before_files = set(raw_dir.glob("*.tar"))
                
                try:
                    download_landsat(aoi_bbox=aoi_bbox, start_date=s_date, end_date=e_date, max_cloud_cover=cloud_cover)
                    after_files = set(raw_dir.glob("*.tar"))
                    new_files = after_files - before_files
                    
                    if new_files:
                        st.success("Download complete!")
                        st.toast("Pushing raw archives to the Refinery Pipeline...")
                        
                        for raw_archive in new_files:
                            with st.spinner(f"Refining COG archive: {raw_archive.name}"):
                                prep = Preprocessor()
                                prep.refinery_pipeline(str(raw_archive))
                                
                        st.success("✅ End-to-End Ingestion & Refinery Sequence Completed. Proceed to Analytics.")
                    else:
                        st.warning("Warning: No new files were resolved. The API returned empty results or payload was cached.")
                except Exception as e:
                    st.error(f"Critical Ingestion Error Mounted: {e}")

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
            
            if st.button("Generate Intelligence"):
                with st.spinner("Processing massive arrays via RAM windowed chunk limits..."):
                    try:
                        session_path = processed_dir / selected_session
                        analyzer = SpectralAnalyzer()
                        analyzer.generate_analytical_cogs(session_path)
                        
                        st.success(f"✅ Intelligence successfully assembled for footprint {selected_session}.")
                        st.toast("Resulting indices compiled to root 'data/results/'.")
                    except Exception as e:
                        st.error(f"Analytical block crash logged: {e}")

    # -----------------
    # MODULE 3: TACTICAL VISUALIZATION
    # -----------------
    elif app_mode == "3. Tactical Visualization":
        st.title("🗺️ Tactical Visualization")
        st.markdown("Use the native Leafmap Layer Control (top right of map) to seamlessly toggle injected intelligence overlays.")
            
        available_sessions = [d.name for d in results_dir.iterdir() if d.is_dir()] if results_dir.exists() else []
        
        if not available_sessions:
            st.info("No generated topology found. Run Analytics algorithms first.")
        else:
            selected_session = st.selectbox("Geospatial Session Index", available_sessions)
            
            active_layers = st.multiselect(
                "Active Layers to Load", 
                ["NDVI", "NDWI", "CLAY", "IRON_OXIDE"], 
                default=["CLAY", "IRON_OXIDE"]
            )
            
            # Map Base instance initialized
            m = leafmap.Map(draw_control=False, measure_control=True, fullscreen_control=True, layer_control=True)
            m.add_basemap("SATELLITE")
            
            cmap_mapping = {
                "NDVI": "RdYlGn",
                "NDWI": "Blues",
                "CLAY": "YlOrBr",      
                "IRON_OXIDE": "Reds"   
            }
            
            session_results = results_dir / selected_session
            rendered_any = False
            
            for layer in active_layers:
                cog_path = session_results / f"{selected_session}_{layer}_COG.tif"
                
                if cog_path.exists():
                    cmap_choice = cmap_mapping.get(layer, "viridis")
                    resolved_posix_path = cog_path.resolve().as_posix()
                    
                    try:
                        m.add_raster(
                            resolved_posix_path,
                            cmap=cmap_choice,
                            layer_name=f"{layer} Signature",
                            nodata=-9999.0
                        )
                        rendered_any = True
                    except Exception as e:
                        st.error(f"Render engine mismatch loading {layer}: {e}")
                else:
                    st.error(f"Layer footprint {layer} unresolved dynamically. Generating intelligence typically solves this.")
                    
            if rendered_any:
                m.to_streamlit(height=700)

except Exception as e:
    # Fail-safe root level UI catcher keeping the client inside the React frontend without a traceback crash loop
    st.error("🚨 Critical Subsystem Failure Intercepted.")
    st.error(f"Details: {e}")
    st.info("Please verify the conda/uv environments and ensure `localtileserver` is installed for Windows.")

