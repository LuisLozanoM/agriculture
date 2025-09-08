import streamlit as st
import datetime as dt
import pandas as pd
import numpy as np
import folium
from streamlit.components.v1 import html as st_html
from typing import Optional
try:
    from geopy.geocoders import Nominatim
except Exception:
    Nominatim = None  # optional dependency; we handle gracefully

try:
    import ee
except Exception as e:  # pragma: no cover
    ee = None


# -----------------------
# Helpers: Earth Engine
# -----------------------
def initialize_ee(project_id: Optional[str] = None) -> None:
    """Authenticate and initialize the Earth Engine client.

    Attempts Initialize first; on failure, runs Authenticate then Initialize.
    """
    if ee is None:
        raise RuntimeError(
            "google-earth-engine (ee) is not installed. Install with: pip install earthengine-api"
        )
    try:
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()
    except Exception:
        ee.Authenticate()
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()


def ee_array_to_df(arr: list, bands: list[str]) -> pd.DataFrame:
    """Convert ee.Image.getRegion output array to a tidy pandas DataFrame.

    Expects bands to be a list of selected band names present in the array.
    """
    df = pd.DataFrame(arr)
    if df.empty:
        return pd.DataFrame(columns=["longitude", "latitude", "time", *bands])

    headers = df.iloc[0]
    df = pd.DataFrame(df.values[1:], columns=headers)

    # Keep only useful columns and drop rows with missing values in bands.
    cols = ["longitude", "latitude", "time", *bands]
    df = df[cols].dropna()

    # Convert types
    df["time"] = pd.to_datetime(df["time"].astype(float), unit="ms")
    for b in bands:
        df[b] = pd.to_numeric(df[b], errors="coerce")
    return df.dropna()


def t_modis_to_celsius(t_modis: pd.Series | np.ndarray | float) -> pd.Series:
    """Convert MODIS LST units to °C (scale 0.02 K then Kelvin->Celsius)."""
    return 0.02 * pd.to_numeric(t_modis) - 273.15


def get_lst_collection(i_date: str, f_date: str, band: str = "LST_Day_1km") -> "ee.ImageCollection":
    lst = (
        ee.ImageCollection("MODIS/061/MOD11A1")
        .select([band])
        .filterDate(i_date, f_date)
    )
    return lst


def get_elevation_image() -> "ee.Image":
    return ee.Image("USGS/SRTMGL1_003")


def get_timeseries_at_point(point: "ee.Geometry", i_date: str, f_date: str, scale_m: int, band: str) -> pd.DataFrame:
    lst = get_lst_collection(i_date, f_date, band)
    # getRegion returns client-side array with columns including time and LST_Day_1km.
    arr = lst.getRegion(point, scale_m).getInfo()
    df = ee_array_to_df(arr, [band])
    if df.empty:
        return df
    df["LST_C"] = t_modis_to_celsius(df[band]).astype(float)
    return df[["time", "LST_C"]].sort_values("time")


def get_point_elevation_m(point: "ee.Geometry", scale_m: int) -> float | None:
    elv = get_elevation_image()
    try:
        val = elv.sample(point, scale_m).first().get("elevation").getInfo()
        return float(val) if val is not None else None
    except Exception:
        return None


def lst_mean_celsius_image(i_date: str, f_date: str, band: str) -> "ee.Image":
    lst_img = get_lst_collection(i_date, f_date, band).mean().select(band)
    # Scale and convert to Celsius
    lst_img = lst_img.multiply(0.02).add(-273.15)
    return lst_img


def ee_thumb_url(image: "ee.Image", region: "ee.Geometry", vis: dict) -> str:
    return image.getThumbURL({**vis, "region": region})


def make_folium_map(lat: float, lon: float, secondary: Optional[tuple[float, float]], buffer_km: float) -> folium.Map:
    m = folium.Map(location=[lat, lon], zoom_start=9, control_scale=True)
    folium.Marker([lat, lon], tooltip="Primary point").add_to(m)
    if secondary is not None:
        folium.Marker([secondary[1], secondary[0]], tooltip="Secondary point", icon=folium.Icon(color="green")).add_to(m)
    folium.Circle([lat, lon], radius=buffer_km * 1000, color="#3186cc", fill=False).add_to(m)
    return m


def add_ee_tile_layer(m: folium.Map, image: "ee.Image", vis: dict, name: str, shown: bool = True, opacity: float = 0.8, region: Optional["ee.Geometry"] = None) -> None:
    """Add an Earth Engine Image as a folium TileLayer.

    Uses Earth Engine map tiles; requires network when running locally.
    """
    if region is not None:
        try:
            image = image.clip(region)
        except Exception:
            pass
    try:
        mapid = image.getMapId(vis)
        # Prefer tile_fetcher if present
        tile_url = mapid.get("tile_fetcher").url_format if mapid.get("tile_fetcher") else f"https://earthengine.googleapis.com/map/{mapid['mapid']}/{{z}}/{{x}}/{{y}}?token={mapid['token']}"
        folium.raster_layers.TileLayer(
            tiles=tile_url,
            name=name,
            attr="Google Earth Engine",
            overlay=True,
            control=True,
            show=shown,
            opacity=opacity,
        ).add_to(m)
    except Exception as e:
        # If EE tiles fail, we simply skip adding the layer; the app shows a warning separately.
        pass


# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Earth Engine LST & Elevation Explorer", layout="wide")
st.title("Google Earth Engine: LST & Elevation Explorer")
st.caption("Enter coordinates and date range to analyze MODIS LST and SRTM elevation. \n"
           "This app uses the Earth Engine Python API.")

with st.sidebar:
    st.header("Inputs")
    default_start = dt.date(2017, 1, 1)
    default_end = dt.date(2020, 1, 1)

    project_id = st.text_input("Earth Engine project (optional)", value="ee-lalozanom")

    # Initialize defaults in session_state so we can programmatically update them BEFORE widgets are created.
    if "lat" not in st.session_state:
        st.session_state["lat"] = 45.7758
    if "lon" not in st.session_state:
        st.session_state["lon"] = 4.8148
    if "sec_lat" not in st.session_state:
        st.session_state["sec_lat"] = 45.574064
    if "sec_lon" not in st.session_state:
        st.session_state["sec_lon"] = 5.175964

    st.subheader("Search a place")
    geocode_query = st.text_input("Address, city, place name", placeholder="e.g. Lyon, France", key="geocode_query")
    apply_target = st.radio("Apply to", ["Primary", "Secondary"], horizontal=True, key="geocode_target")

    if st.button("Geocode"):
        if not geocode_query.strip():
            st.warning("Enter a place to search.")
        elif Nominatim is None:
            st.error("Geocoding requires geopy. Install with: pip install geopy")
        else:
            try:
                geolocator = Nominatim(user_agent="st_gee_app")
                results = geolocator.geocode(geocode_query.strip(), exactly_one=False, limit=5, addressdetails=True)
                if not results:
                    st.warning("No results found for that query.")
                else:
                    st.session_state["geocode_results"] = [
                        {"address": r.address, "lat": r.latitude, "lon": r.longitude} for r in results
                    ]
                    # Do not auto-apply; allow user to select and apply explicitly
                    st.rerun()
            except Exception as e:
                st.error(f"Geocoding failed: {e}")

    # If we have results, let user refine choice
    if isinstance(st.session_state.get("geocode_results"), list) and st.session_state["geocode_results"]:
        opts = [
            f"{i+1}. {r['address']} ({r['lat']:.5f}, {r['lon']:.5f})" for i, r in enumerate(st.session_state["geocode_results"])
        ]
        sel_idx = st.selectbox("Select a result", options=list(range(len(opts))), format_func=lambda i: opts[i], key="geocode_choice")
        if st.button("Apply selected"):
            chosen = st.session_state["geocode_results"][sel_idx]
            if apply_target == "Primary":
                st.session_state["lat"] = float(chosen["lat"]) 
                st.session_state["lon"] = float(chosen["lon"]) 
            else:
                st.session_state["sec_lat"] = float(chosen["lat"]) 
                st.session_state["sec_lon"] = float(chosen["lon"]) 
            st.rerun()

    st.divider()
    # Primary coordinates
    lat = st.number_input("Latitude (primary)", value=st.session_state["lat"], format="%.6f", key="lat")
    lon = st.number_input("Longitude (primary)", value=st.session_state["lon"], format="%.6f", key="lon")

    add_secondary = st.checkbox("Add secondary point (compare)", value=True)
    sec_lat, sec_lon = None, None
    if add_secondary:
        sec_lat = st.number_input("Latitude (secondary)", value=st.session_state["sec_lat"], format="%.6f", key="sec_lat")
        sec_lon = st.number_input("Longitude (secondary)", value=st.session_state["sec_lon"], format="%.6f", key="sec_lon")

    col1, col2 = st.columns(2)
    with col1:
        i_date = st.date_input("Start date", value=default_start)
    with col2:
        f_date = st.date_input("End date (exclusive)", value=default_end)

    scale_m = st.number_input("Sampling scale (meters)", min_value=250, max_value=5000, value=1000, step=250)
    buffer_km = st.number_input("ROI buffer (km)", min_value=10.0, max_value=2000.0, value=1000.0, step=10.0)

    vis_lst = {
        "min": st.number_input("LST min (°C)", value=10.0, step=1.0),
        "max": st.number_input("LST max (°C)", value=30.0, step=1.0),
        "palette": ["blue", "yellow", "orange", "red"],
        "dimensions": 768,
    }

    vis_elv = {
        "min": 0,
        "max": st.number_input("Elevation max (m)", value=2000, step=100),
        "palette": ["006633", "E5FFCC", "662A00", "D8D8D8", "F5F5F5"],
        "dimensions": 768,
    }

    st.header("LST Options")
    tod_choice = st.radio("LST time of day", ["Daytime", "Nighttime", "Both"], horizontal=True)
    is_both = tod_choice == "Both"
    lst_band = "LST_Day_1km" if tod_choice == "Daytime" else ("LST_Night_1km" if tod_choice == "Nighttime" else None)

    st.header("Map Layers")
    day_name = "LST Day (mean °C)"
    night_name = "LST Night (mean °C)"
    if is_both:
        layer_options = [day_name, night_name, "Elevation (SRTM)"]
        default_layers = [day_name]
    else:
        lst_layer_name = day_name if tod_choice == "Daytime" else night_name
        layer_options = [lst_layer_name, "Elevation (SRTM)"]
        default_layers = [lst_layer_name]
    selected_layers = st.multiselect("Select layers to show on the map", options=layer_options, default=default_layers)
    layer_opacity = st.slider("Layer opacity", 0.1, 1.0, 0.8, 0.05)

    run = st.button("Run analysis")


if run:
    # Initialize Earth Engine
    try:
        initialize_ee(project_id.strip() or None)
    except Exception as e:
        st.error(f"Failed to initialize Earth Engine: {e}")
        st.stop()

    # Geometries
    primary_point = ee.Geometry.Point(float(lon), float(lat))
    secondary_point = None
    if add_secondary and sec_lat is not None and sec_lon is not None:
        secondary_point = ee.Geometry.Point(float(sec_lon), float(sec_lat))

    # Elevation at points
    with st.spinner("Sampling elevation and LST time series..."):
        elv_primary = get_point_elevation_m(primary_point, int(scale_m))
        elv_secondary = None
        if secondary_point is not None:
            elv_secondary = get_point_elevation_m(secondary_point, int(scale_m))

        # Time series
        start_str = i_date.strftime("%Y-%m-%d")
        end_str = f_date.strftime("%Y-%m-%d")
        if is_both:
            # Daytime
            ts_primary_day = get_timeseries_at_point(primary_point, start_str, end_str, int(scale_m), "LST_Day_1km")
            ts_secondary_day = pd.DataFrame()
            if secondary_point is not None:
                ts_secondary_day = get_timeseries_at_point(secondary_point, start_str, end_str, int(scale_m), "LST_Day_1km")
            # Nighttime
            ts_primary_night = get_timeseries_at_point(primary_point, start_str, end_str, int(scale_m), "LST_Night_1km")
            ts_secondary_night = pd.DataFrame()
            if secondary_point is not None:
                ts_secondary_night = get_timeseries_at_point(secondary_point, start_str, end_str, int(scale_m), "LST_Night_1km")
        else:
            ts_primary = get_timeseries_at_point(primary_point, start_str, end_str, int(scale_m), lst_band)
            ts_secondary = pd.DataFrame()
            if secondary_point is not None:
                ts_secondary = get_timeseries_at_point(secondary_point, start_str, end_str, int(scale_m), lst_band)

    # Point info
    st.subheader("Point Information")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Primary point**")
        st.write({"lat": float(lat), "lon": float(lon), "elevation_m": elv_primary})
    with cols[1]:
        if secondary_point is not None:
            st.markdown("**Secondary point**")
            st.write({"lat": float(sec_lat), "lon": float(sec_lon), "elevation_m": elv_secondary})
        else:
            st.empty()

    # Time series charts
    st.subheader("LST Time Series (°C)")
    if is_both:
        # Day chart
        st.markdown("Daytime")
        if ts_primary_day.empty and (secondary_point is None or ts_secondary_day.empty):
            st.warning("No daytime LST data for the selected period/points.")
        else:
            df_p = pd.DataFrame()
            if not ts_primary_day.empty:
                df_p = ts_primary_day.set_index("time").rename(columns={"LST_C": "Primary (Day)"})
            df_s = pd.DataFrame()
            if secondary_point is not None and not ts_secondary_day.empty:
                df_s = ts_secondary_day.set_index("time").rename(columns={"LST_C": "Secondary (Day)"})
            ts_plot_day = df_p.join(df_s, how="outer").sort_index() if not df_p.empty or not df_s.empty else pd.DataFrame()
            if not ts_plot_day.empty:
                st.scatter_chart(ts_plot_day)

        # Night chart
        st.markdown("Nighttime")
        if ts_primary_night.empty and (secondary_point is None or ts_secondary_night.empty):
            st.warning("No nighttime LST data for the selected period/points.")
        else:
            df_pn = pd.DataFrame()
            if not ts_primary_night.empty:
                df_pn = ts_primary_night.set_index("time").rename(columns={"LST_C": "Primary (Night)"})
            df_sn = pd.DataFrame()
            if secondary_point is not None and not ts_secondary_night.empty:
                df_sn = ts_secondary_night.set_index("time").rename(columns={"LST_C": "Secondary (Night)"})
            ts_plot_night = df_pn.join(df_sn, how="outer").sort_index() if not df_pn.empty or not df_sn.empty else pd.DataFrame()
            if not ts_plot_night.empty:
                st.scatter_chart(ts_plot_night)
    else:
        if ts_primary.empty:
            st.warning("No LST data returned for the primary point and date range.")
        else:
            ts_primary_idx = ts_primary.set_index("time").rename(columns={"LST_C": "Primary"})
            if not ts_secondary.empty:
                ts_secondary_idx = ts_secondary.set_index("time").rename(columns={"LST_C": "Secondary"})
                ts_plot = ts_primary_idx.join(ts_secondary_idx, how="outer").sort_index()
            else:
                ts_plot = ts_primary_idx
            st.scatter_chart(ts_plot)

    # ROI and thumbnails
    st.subheader("Thumbnails (ROI around primary point)")
    roi = primary_point.buffer(float(buffer_km) * 1000.0)

    if is_both:
        lst_img_day = lst_mean_celsius_image(start_str, end_str, "LST_Day_1km")
        lst_img_night = lst_mean_celsius_image(start_str, end_str, "LST_Night_1km")
    else:
        lst_img = lst_mean_celsius_image(start_str, end_str, lst_band)
    elv_img = get_elevation_image().updateMask(get_elevation_image().gt(0))

    try:
        if is_both:
            lst_url_d = ee_thumb_url(lst_img_day, roi, vis_lst)
            st.image(lst_url_d, caption="Mean Daytime LST (°C)")
            lst_url_n = ee_thumb_url(lst_img_night, roi, vis_lst)
            st.image(lst_url_n, caption="Mean Nighttime LST (°C)")
        else:
            lst_url = ee_thumb_url(lst_img, roi, vis_lst)
            st.image(lst_url, caption=f"Mean {tod_choice} LST (°C)")
    except Exception as e:
        st.warning(f"Could not fetch LST thumbnail: {e}")

    try:
        elv_url = ee_thumb_url(elv_img, roi, vis_elv)
        st.image(elv_url, caption="Elevation (masked below sea level)")
    except Exception as e:
        st.warning(f"Could not fetch Elevation thumbnail: {e}")

    # Folium map embed
    st.subheader("Map")
    folium_map = make_folium_map(float(lat), float(lon),
                                 (float(sec_lon), float(sec_lat)) if secondary_point is not None else None,
                                 float(buffer_km))

    # Add EE layers to folium map based on selection
    try:
        if is_both:
            if "LST Day (mean °C)" in selected_layers:
                add_ee_tile_layer(folium_map, lst_img_day, {k: v for k, v in vis_lst.items() if k != "dimensions"},
                                  name="LST Day (mean °C)", shown=True, opacity=float(layer_opacity), region=roi)
            if "LST Night (mean °C)" in selected_layers:
                add_ee_tile_layer(folium_map, lst_img_night, {k: v for k, v in vis_lst.items() if k != "dimensions"},
                                  name="LST Night (mean °C)", shown="LST Day (mean °C)" not in selected_layers, opacity=float(layer_opacity), region=roi)
            if "Elevation (SRTM)" in selected_layers:
                add_ee_tile_layer(folium_map, elv_img, {k: v for k, v in vis_elv.items() if k != "dimensions"},
                                  name="Elevation (SRTM)", shown=False, opacity=float(layer_opacity), region=roi)
        else:
            if lst_layer_name in selected_layers:
                add_ee_tile_layer(folium_map, lst_img, {k: v for k, v in vis_lst.items() if k != "dimensions"},
                                  name=lst_layer_name, shown=True, opacity=float(layer_opacity), region=roi)
            if "Elevation (SRTM)" in selected_layers:
                add_ee_tile_layer(folium_map, elv_img, {k: v for k, v in vis_elv.items() if k != "dimensions"},
                                  name="Elevation (SRTM)", shown=lst_layer_name not in selected_layers, opacity=float(layer_opacity), region=roi)
        folium.LayerControl(collapsed=False).add_to(folium_map)
    except Exception as e:
        st.warning(f"Could not add EE tile layers to the map: {e}")

    # Render Folium map in Streamlit without extra dependencies
    m_html = folium_map.get_root().render()
    st_html(m_html, height=520)

else:
    st.info("Set inputs in the sidebar and click 'Run analysis'.")
