"""
TerraForge Mining Intelligence - Ingestion Pipeline
Multi-Satellite Data Downloader

Downloads Level-2A from Copernicus using Sentinelsat.
Downloads Landsat using native requests to USGS M2M API with strict headers and session handling.
"""

import argparse
import os
import json
import logging
import time
import requests
from dotenv import load_dotenv

# Sentinel specific
from sentinelsat import SentinelAPI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataDownloader")

def download_sentinel(aoi_wkt: str, start_date: str, end_date: str, cloud_cover: tuple = (0, 20)):
    """Handles Sentinel-2 downloads via Copernicus Data Space Ecosystem."""
    load_dotenv()
    user = os.environ.get('COPERNICUS_USER')
    password = os.environ.get('COPERNICUS_PASSWORD')
    
    if not user or not password:
        logger.error("Environment variables COPERNICUS_USER and COPERNICUS_PASSWORD must be set in .env")
        return

    cdse_url = "https://catalogue.dataspace.copernicus.eu/odata/v1"
    try:
        api = SentinelAPI(user, password, cdse_url)
        logger.info("Successfully authenticated with Copernicus (CDSE).")
    except Exception as e:
        logger.error(f"Failed to connect to Copernicus API: {e}")
        return

    logger.info(f"Querying Copernicus Catalogue for Sentinel-2 L2A images ({start_date} to {end_date})...")
    try:
        products = api.query(
            aoi_wkt,
            date=(start_date, end_date),
            platformname='Sentinel-2',
            processinglevel='Level-2A',
            cloudcoverpercentage=cloud_cover
        )
    except Exception as e:
        logger.error(f"Failed to execute Copernicus spatial query: {e}")
        return

    if not products:
        logger.info("ℹ️ No Sentinel products found for the specified AOI and timeframe.")
        return

    try:
        products_df = api.to_dataframe(products)
        products_df_sorted = products_df.sort_values(by="cloudcoverpercentage")
        best_product = products_df_sorted.head(1).index[0]
        logger.info(f"Sentinel-2: Best product selected: {products_df_sorted.loc[best_product, 'title']}")
    except Exception as e:
        logger.error(f"Error processing Sentinel metadata DataFrame: {e}")
        return

    download_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "sentinel")
    os.makedirs(download_dir, exist_ok=True)
    
    try:
        logger.info(f"Downloading Sentinel product to: {download_dir}")
        api.download(best_product, directory_path=download_dir)
        logger.info("✅ Sentinel download completed successfully.")
    except KeyboardInterrupt:
        logger.warning("Download interrupted by user.")
    except Exception as e:
        logger.error(f"Failed to download Sentinel product: {e}")
        logger.info("Retrying in 10 seconds...")
        time.sleep(10)
        try:
            api.download(best_product, directory_path=download_dir)
            logger.info("✅ Retry download completed successfully.")
        except Exception as retry_e:
            logger.error(f"Retry Sentinel download failed: {retry_e}")


def download_landsat(aoi_bbox: tuple, start_date: str, end_date: str, max_cloud_cover: int = 10):
    """Handles Landsat downloads via USGS M2M API using requests.Session() and dynamic auth fallback."""
    load_dotenv()
    username = os.environ.get('USGS_USER')
    usgs_pat = os.environ.get('USGS_PAT')
    
    if not username or not usgs_pat:
        logger.error("Environment variables USGS_USER and USGS_PAT must be set in .env")
        raise ValueError("Missing USGS_USER or USGS_PAT in environment.")

    base_url = "https://m2m.cr.usgs.gov/api/api/json/stable/"
    
    # Initialize Session
    session = requests.Session()
    session.headers.update({
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"
    })
    
    logger.info("Authenticating with USGS M2M API...")
    
    api_key = None
    
    # 1. Try login-token
    login_token_url = f"{base_url}login-token"
    logger.info(f"Trying Primary Auth: {login_token_url}")
    try:
        r = session.post(login_token_url, json={"username": username, "token": usgs_pat})
        if r.status_code == 200:
            data = r.json()
            if not data.get("errorCode"):
                api_key = data.get("data")
                logger.info("✅ Authenticated via /login-token successfully.")
            else:
                logger.warning(f"/login-token logic error: {data.get('errorMessage')}")
        else:
            logger.warning(f"HTTP {r.status_code} received from {login_token_url}")
            logger.debug(f"Response: {r.text}")
    except Exception as e:
        logger.warning(f"Network error on {login_token_url}: {e}")

    # 2. Fallback to /login
    if not api_key:
        login_legacy_url = f"{base_url}login"
        logger.info(f"Fallback Auth: Trying {login_legacy_url}")
        try:
            r = session.post(login_legacy_url, json={"username": username, "password": usgs_pat})
            if r.status_code == 200:
                data = r.json()
                if not data.get("errorCode"):
                    api_key = data.get("data")
                    logger.info("✅ Authenticated via /login successfully.")
                else:
                    logger.error(f"Fallback /login logic error: {data.get('errorMessage')}")
                    return
            else:
                logger.error(f"HTTP {r.status_code} received from {login_legacy_url}")
                logger.error(f"Response: {r.text}")
                return
        except Exception as e:
            logger.error(f"Network error on {login_legacy_url}: {e}")
            return

    if not api_key:
        logger.error("Authentication failed. Ensure your USGS_PAT or Credentials are correct and active.")
        return

    # Update session headers with Auth Token
    session.headers.update({"X-Auth-Token": api_key})
    
    # 3. Search for scenes
    search_url = f"{base_url}scene-search"
    search_payload = {
        "datasetName": "landsat_ot_c2_l2",
        "sceneFilter": {
            "spatialFilter": {
                "filterType": "mbr",
                "lowerLeft": {"latitude": aoi_bbox[1], "longitude": aoi_bbox[0]},
                "upperRight": {"latitude": aoi_bbox[3], "longitude": aoi_bbox[2]}
            },
            "acquisitionFilter": {
                "start": f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}",
                "end": f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
            },
            "cloudCoverFilter": {
                "max": max_cloud_cover,
                "min": 0,
                "includeUnknown": False
            }
        },
        "maxResults": 10
    }
    
    try:
        logger.info(f"Executing scene extraction on confirmed endpoint: {search_url}")
        r_search = session.post(search_url, json=search_payload)
        
        if r_search.status_code != 200:
            logger.error(f"USGS API error HTTP {r_search.status_code}")
            logger.error(f"Headers: {r_search.headers}")
            logger.error(f"Raw: {r_search.text}")
            return
            
        search_data = r_search.json()
        
        if search_data.get("errorCode"):
            logger.error(f"USGS Application API Error: {search_data.get('errorMessage')}")
            return
            
        results = search_data.get("data", {}).get("results", [])
        if not results:
            logger.info("ℹ️ No Landsat products found for the specified AOI and timeframe. Adjust filters.")
            return

        # Sort scenes by cloud cover
        results = sorted(results, key=lambda x: x.get('cloudCover', 100))
        best_scene = results[0]
        scene_id = best_scene.get('entityId')
        display_id = best_scene.get('displayId', scene_id)
        
        logger.info(f"🔎 Landsat Traceability: Found Top Product: {display_id} (Cloud cover: {best_scene.get('cloudCover')}%)")
        
        download_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "landsat")
        os.makedirs(download_dir, exist_ok=True)
        
        # Save metadata to JSON trace file
        metadata_path = os.path.join(download_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(best_scene, f, indent=4)
        logger.info(f"Metadata securely logged to {metadata_path}")
        
        # Get download options
        options_url = f"{base_url}download-options"
        options_payload = {
            "datasetName": "landsat_ot_c2_l2",
            "entityIds": [scene_id]
        }
        
        r_opt = session.post(options_url, json=options_payload)
        if r_opt.status_code != 200:
            logger.error(f"Failed to fetch download-options. HTTP {r_opt.status_code}")
            logger.error(r_opt.text)
            return
            
        opt_data = r_opt.json().get("data", [])
        
        products = []
        for opt in opt_data:
            if opt.get("available") and opt.get("id"):
                 products.append({"entityId": opt["entityId"], "productId": opt["id"]})
        
        if not products:
             logger.error("No standard product payload was found available to request for this scene.")
             return
             
        # Request actual download URL
        dl_url = f"{base_url}download-request"
        dl_payload = {
            "downloads": products,
            "label": "TerraForge_Ingestion"
        }
        
        logger.info("Executing USGS /download-request endpoint...")
        r_dl = session.post(dl_url, json=dl_payload)
        if r_dl.status_code != 200:
            logger.error(f"Failed to submit download-request. HTTP {r_dl.status_code}")
            logger.error(r_dl.text)
            return
            
        dl_data = r_dl.json()
        
        available_downloads = dl_data.get("data", {}).get("availableDownloads", [])
        if not available_downloads:
             logger.error(f"Download request failed or scene is not immediately available. Trace: {dl_data}")
             return
             
        actual_url = available_downloads[0].get("url")
        logger.info(f"✅ Intercepted USGS payload URL. Proceeding to fetch bundle...")
        
        bundle_path = os.path.join(download_dir, f"{display_id}.tar")
        with session.get(actual_url, stream=True) as d_req:
             d_req.raise_for_status()
             with open(bundle_path, 'wb') as f:
                 for chunk in d_req.iter_content(chunk_size=1024*1024):
                     f.write(chunk)
                     
        logger.info(f"✅ Landsat download completed successfully: {bundle_path}")

    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP Connection Error to USGS M2M API: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during direct Landsat M2M download: {e}")
    finally:
        # Gracefully log out to release the token from USGS servers
        if api_key:
            try:
                logout_url = f"{base_url}logout"
                session.post(logout_url)
                logger.info("Session token legitimately destroyed.")
            except:
                pass


def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="TerraForge Mining Intelligence - Ingestion Downloader")
    parser.add_argument("--satellite", type=str, choices=["sentinel", "landsat"], required=True,
                        help="Select satellite source: 'sentinel' or 'landsat'.")
    parser.add_argument("--min_lon", type=float, required=True)
    parser.add_argument("--min_lat", type=float, required=True)
    parser.add_argument("--max_lon", type=float, required=True)
    parser.add_argument("--max_lat", type=float, required=True)
    parser.add_argument("--start_date", type=str, required=True)
    parser.add_argument("--end_date", type=str, required=True)
    args = parser.parse_args()
    
    target_satellite = args.satellite.lower()
    aoi_bbox = (args.min_lon, args.min_lat, args.max_lon, args.max_lat)

    logger.info(f"🚀 *** TerraForge Ingestion Engine - Target: {target_satellite.upper()} ***")

    if target_satellite == "sentinel":
        # wkt derived statically just for standard fallbacks here if ever triggered directly by shell
        aoi_wkt = f"POLYGON(({args.min_lon} {args.min_lat}, {args.max_lon} {args.min_lat}, {args.max_lon} {args.max_lat}, {args.min_lon} {args.max_lat}, {args.min_lon} {args.min_lat}))"
        download_sentinel(aoi_wkt, args.start_date, args.end_date)
    elif target_satellite == "landsat":
        download_landsat(aoi_bbox, args.start_date, args.end_date)


if __name__ == "__main__":
    main()
