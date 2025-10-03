import streamlit as st
import ee
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import requests
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Page config
st.set_page_config(page_title="Earth Engine Chat Assistant", layout="wide", page_icon="🌍")

# Initialize Earth Engine
@st.cache_resource
def initialize_earth_engine():
    try:
        ee.Initialize(project='ee-lalozanom')
    except Exception:
        ee.Authenticate()
        ee.Initialize(project='ee-lalozanom')

initialize_earth_engine()

# Helper functions from your notebook
def ee_array_to_df(arr, list_of_bands):
    """Transforms client-side ee.Image.getRegion array to pandas.DataFrame."""
    df = pd.DataFrame(arr)
    headers = df.iloc[0]
    df = pd.DataFrame(df.values[1:], columns=headers)
    df = df[['longitude', 'latitude', 'time', *list_of_bands]].dropna()
    for band in list_of_bands:
        df[band] = pd.to_numeric(df[band], errors='coerce')
    df['datetime'] = pd.to_datetime(df['time'], unit='ms')
    df = df[['time','datetime',  *list_of_bands]]
    return df

def t_modis_to_celsius(t_modis):
    """Converts MODIS LST units to degrees Celsius."""
    return 0.02*t_modis - 273.15

def add_ee_layer(self, ee_image_object, vis_params, name):
    """Adds a method for displaying Earth Engine image tiles to folium map."""
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
        name=name,
        overlay=True,
        control=True
    ).add_to(self)

folium.Map.add_ee_layer = add_ee_layer

# Define available tools for the LLM
AVAILABLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_earth_engine_visualization",
            "description": "Generate Earth Engine visualizations (maps, plots, data) for a specific location. Use this when user asks to see maps, temperature data, elevation, land cover, or any Earth Engine data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city, place name, or coordinates (e.g., 'Lyon', 'Mexico City', '45.77, 4.85')"
                    },
                    "visualizations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of visualizations to generate. Options: 'daytime_lst', 'lst_map', 'land_cover', 'elevation', 'all'"
                    },
                    "start_date": {
                        "type": "string",
                        "description": "Start date in YYYY-MM-DD format (optional, defaults to 3 years ago)"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date in YYYY-MM-DD format (optional, defaults to today)"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# Ollama interaction with function calling
def call_ollama_with_tools(messages: List[Dict], model="gpt-oss:20b", tools=None):
    """Call Ollama API with function calling support."""
    url = "http://localhost:11434/api/chat"
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.7
        }
    }
    
    if tools:
        payload["tools"] = tools
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Could not connect to Ollama. Make sure Ollama is running (ollama serve)."}
    except Exception as e:
        return {"error": f"Error calling Ollama: {str(e)}"}

def call_ollama(prompt, model="gpt-oss:20b", system_message=None):
    """Simple Ollama call without tools."""
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})
    
    result = call_ollama_with_tools(messages, model)
    if "error" in result:
        return result["error"]
    return result.get("message", {}).get("content", "No response")

# Note: extract_location_info removed - now using function calling instead

def geocode_location(location_name):
    """Simple geocoding using Nominatim with retry logic."""
    try:
        from geopy.geocoders import Nominatim
        from geopy.exc import GeocoderTimedOut, GeocoderServiceError
        
        geolocator = Nominatim(user_agent="earth_engine_chat_app", timeout=10)
        
        # Try with original name
        try:
            location = geolocator.geocode(location_name, language='en')
            if location:
                return location.latitude, location.longitude
        except (GeocoderTimedOut, GeocoderServiceError):
            pass
        
        # Try with country appended if not already there
        if ',' not in location_name:
            try:
                location = geolocator.geocode(f"{location_name}, World", language='en')
                if location:
                    return location.latitude, location.longitude
            except (GeocoderTimedOut, GeocoderServiceError):
                pass
                
    except Exception as e:
        st.warning(f"Geocoding error: {str(e)}")
    
    return None, None

# Earth Engine processing functions
def get_lst_data(lat, lon, i_date, f_date, scale=1000):
    """Get LST time series data for a point."""
    poi = ee.Geometry.Point(lon, lat)
    lst = ee.ImageCollection('MODIS/061/MOD11A1').select('LST_Day_1km', 'QC_Day').filterDate(i_date, f_date)
    
    lst_poi = lst.getRegion(poi, scale).getInfo()
    lst_df = ee_array_to_df(lst_poi, ['LST_Day_1km'])
    lst_df['LST_Day_1km'] = lst_df['LST_Day_1km'].apply(t_modis_to_celsius)
    
    return lst_df

def create_lst_plot(lst_df, location_name):
    """Create LST time series plot with fitted curve."""
    x_data = np.asanyarray(lst_df['time'].apply(float))
    y_data = np.asanyarray(lst_df['LST_Day_1km'].apply(float))
    
    def fit_func(t, lst0, delta_lst, tau, phi):
        return lst0 + (delta_lst/2)*np.sin(2*np.pi*t/tau + phi)
    
    lst0 = 20
    delta_lst = 40
    tau = 365*24*3600*1000
    phi = 2*np.pi*4*30.5*3600*1000/tau
    
    try:
        params, _ = optimize.curve_fit(fit_func, x_data, y_data, p0=[lst0, delta_lst, tau, phi])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(lst_df['datetime'], lst_df['LST_Day_1km'], c='blue', alpha=0.5, label='Data')
        ax.plot(lst_df['datetime'], fit_func(x_data, *params), label='Fitted', color='red', lw=2.5)
        ax.set_title(f'Daytime Land Surface Temperature - {location_name}', fontsize=16)
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Temperature [°C]', fontsize=14)
        ax.grid(lw=0.2)
        ax.legend(fontsize=12)
        
        return fig
    except:
        return None

def create_ee_map(lat, lon, visualizations, i_date, f_date):
    """Create folium map with requested Earth Engine layers."""
    poi = ee.Geometry.Point(lon, lat)
    roi = poi.buffer(1e6)
    
    my_map = folium.Map(location=[lat, lon], zoom_start=8)
    
    # Add marker for location
    folium.Marker([lat, lon], popup="Selected Location", icon=folium.Icon(color='red')).add_to(my_map)
    
    try:
        # LST Map
        if 'lst_map' in visualizations or 'all' in visualizations:
            lst = ee.ImageCollection('MODIS/061/MOD11A1').select('LST_Day_1km').filterDate(i_date, f_date)
            lst_img = lst.mean().multiply(0.02).add(-273.15)
            lst_vis = {'min': 0, 'max': 40, 'palette': ['white', 'blue', 'green', 'yellow', 'orange', 'red']}
            my_map.add_ee_layer(lst_img, lst_vis, 'Land Surface Temperature')
    except Exception as e:
        st.warning(f"Could not load LST layer: {str(e)}")
    
    try:
        # Elevation
        if 'elevation' in visualizations or 'all' in visualizations:
            elv = ee.Image('USGS/SRTMGL1_003')
            elv_masked = elv.updateMask(elv.gt(0))
            elv_vis = {'min': 0, 'max': 4000, 'palette': ['006633', 'E5FFCC', '662A00', 'D8D8D8', 'F5F5F5']}
            my_map.add_ee_layer(elv_masked, elv_vis, 'Elevation')
    except Exception as e:
        st.warning(f"Could not load elevation layer: {str(e)}")
    
    try:
        # Land Cover
        if 'land_cover' in visualizations or 'all' in visualizations:
            lc = ee.ImageCollection('MODIS/061/MCD12Q1').select('LC_Type1')
            # Use a specific date for land cover
            lc_img = lc.filterDate(i_date, f_date).first()
            # If no image in date range, get the most recent
            if lc_img is None:
                lc_img = lc.sort('system:time_start', False).first()
            
            lc_vis = {
                'min': 1, 'max': 17,
                'palette': ['05450a','086a10', '54a708', '78d203', '009900', 'c6b044',
                           'dcd159', 'dade48', 'fbff13', 'b6ff05', '27ff87', 'c24f44',
                           'a5a5a5', 'ff6d4c', '69fff8', 'f9ffa4', '1c0dff']
            }
            my_map.add_ee_layer(lc_img, lc_vis, 'Land Cover')
    except Exception as e:
        st.warning(f"Could not load land cover layer: {str(e)}")
    
    folium.LayerControl(collapsed=False).add_to(my_map)
    
    return my_map

# Tool execution function (must be defined BEFORE sidebar)
def execute_earth_engine_tool(location: str, visualizations: List[str] = None, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
    """Execute the Earth Engine visualization tool."""
    if visualizations is None:
        visualizations = ['all']
    
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=3*365)).strftime("%Y-%m-%d")
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Geocode location
    lat, lon = geocode_location(location)
    if lat is None:
        return {"error": f"Could not find location: {location}. Please try with a different location name or provide coordinates.", "success": False}
    
    try:
        result = {
            "location": location,
            "lat": lat,
            "lon": lon,
            "start_date": start_date,
            "end_date": end_date,
            "visualizations": visualizations,
            "success": True
        }
        
        # Get LST data if requested
        if 'daytime_lst' in visualizations or 'all' in visualizations:
            lst_df = get_lst_data(lat, lon, start_date, end_date)
            if not lst_df.empty:
                result["lst_data"] = lst_df
                result["lst_plot"] = create_lst_plot(lst_df, location)
        
        # Create thumbnail URLs
        result["thumbnail_urls"] = {}
        poi = ee.Geometry.Point(lon, lat)
        roi = poi.buffer(100000)  # 100km buffer for thumbnails
        
        if 'lst_map' in visualizations or 'all' in visualizations:
            try:
                lst_col = ee.ImageCollection('MODIS/061/MOD11A1').select('LST_Day_1km').filterDate(start_date, end_date)
                lst_img = lst_col.mean().multiply(0.02).add(-273.15)
                lst_vis = {'min': 0, 'max': 40, 'palette': ['white', 'blue', 'green', 'yellow', 'orange', 'red'], 'dimensions': 512}
                result["thumbnail_urls"]["lst"] = lst_img.getThumbURL({**lst_vis, 'region': roi})
            except Exception as e:
                print(f"LST thumbnail error: {e}")
        
        if 'elevation' in visualizations or 'all' in visualizations:
            try:
                elv_img = ee.Image('USGS/SRTMGL1_003').updateMask(ee.Image('USGS/SRTMGL1_003').gt(0))
                elv_vis = {'min': 0, 'max': 4000, 'palette': ['006633', 'E5FFCC', '662A00', 'D8D8D8', 'F5F5F5'], 'dimensions': 512}
                result["thumbnail_urls"]["elevation"] = elv_img.getThumbURL({**elv_vis, 'region': roi})
            except Exception as e:
                print(f"Elevation thumbnail error: {e}")
        
        if 'land_cover' in visualizations or 'all' in visualizations:
            try:
                lc_col = ee.ImageCollection('MODIS/061/MCD12Q1').select('LC_Type1')
                lc_img = lc_col.filterDate(start_date, end_date).first()
                if lc_img is None:
                    lc_img = lc_col.sort('system:time_start', False).first()
                lc_vis = {
                    'min': 1, 'max': 17, 'dimensions': 512,
                    'palette': ['05450a','086a10', '54a708', '78d203', '009900', 'c6b044',
                               'dcd159', 'dade48', 'fbff13', 'b6ff05', '27ff87', 'c24f44',
                               'a5a5a5', 'ff6d4c', '69fff8', 'f9ffa4', '1c0dff']
                }
                result["thumbnail_urls"]["land_cover"] = lc_img.getThumbURL({**lc_vis, 'region': roi})
            except Exception as e:
                print(f"Land cover thumbnail error: {e}")
        
        return result
    except Exception as e:
        return {"error": f"Error generating visualizations: {str(e)}", "success": False}

# Streamlit UI
st.title("🌍 Earth Engine Chat Assistant")
st.caption("Chat naturally to explore Earth Engine data - powered by Ollama")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    ollama_model = st.selectbox(
        "Ollama Model",
        ["gpt-oss:20b", "qwen3:32b"],
        help="Select the Ollama model for chat"
    )
    
    debug_mode = st.checkbox("Debug Mode", value=False, help="Show Ollama responses for debugging")
    
    st.divider()
    
    st.subheader("🚀 Quick Actions")
    st.markdown("**Manual Visualization** (if chatbot doesn't auto-generate):")
    
    quick_location = st.text_input("Location:", placeholder="e.g., Mexico City")
    quick_viz = st.multiselect(
        "Visualizations:",
        ["all", "lst_map", "daytime_lst", "elevation", "land_cover"],
        default=["all"]
    )
    
    if st.button("🗺️ Generate Now", type="primary"):
        if quick_location:
            with st.spinner("Generating visualizations..."):
                result = execute_earth_engine_tool(quick_location, quick_viz)
                if result.get("success"):
                    # Store the generation parameters instead of the objects
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"📍 **Location**: {result['location']} ({result['lat']:.4f}, {result['lon']:.4f})\n📅 **Date Range**: {result['start_date']} to {result['end_date']}\n🗺️ **Visualizations**: {', '.join(result['visualizations'])}",
                        "viz_params": {
                            "location": result["location"],
                            "lat": result["lat"],
                            "lon": result["lon"],
                            "start_date": result["start_date"],
                            "end_date": result["end_date"],
                            "visualizations": result["visualizations"],
                            "thumbnail_urls": result.get("thumbnail_urls", {})
                        }
                    })
                    st.rerun()
                else:
                    st.error(result.get("error", "Failed to generate"))
        else:
            st.warning("Please enter a location")
    
    st.divider()
    st.subheader("What can I do?")
    st.markdown("""
    💬 **Have a conversation** - Ask me anything!
    
    🗺️ **Generate visualizations** - I can show you:
    - Temperature maps and time series
    - Elevation and terrain data
    - Land cover maps
    - Any Earth Engine data for any location
    
    **Try asking:**
    - "What can you help me with?"
    - "Show me the temperature map for Lyon"
    - "Can you compare elevation in Mexico City vs Paris?"
    - "I'm interested in land cover data for my city"
    """)

# System message for the assistant
SYSTEM_MESSAGE = """You are a helpful Earth Engine data assistant. You can have natural conversations with users AND generate Earth Engine visualizations.

When users ask about:
- Temperature, LST (Land Surface Temperature), climate data
- Maps, satellite imagery
- Elevation, terrain, topography
- Land cover, land use
- Any location-specific Earth Engine data

You should use the 'get_earth_engine_visualization' tool to generate the requested visualizations.

Be conversational and friendly. Ask clarifying questions if needed. Explain what you're showing them."""

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = [
        {"role": "system", "content": SYSTEM_MESSAGE}
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # If message has visualization parameters, regenerate the visualizations
        if "viz_params" in message:
            params = message["viz_params"]
            
            # Show LST plot if requested
            if 'daytime_lst' in params["visualizations"] or 'all' in params["visualizations"]:
                try:
                    st.subheader("📈 Daytime LST Time Series")
                    lst_df = get_lst_data(params["lat"], params["lon"], params["start_date"], params["end_date"])
                    if not lst_df.empty:
                        fig = create_lst_plot(lst_df, params["location"])
                        if fig:
                            st.pyplot(fig)
                            with st.expander("View Data Sample"):
                                st.dataframe(lst_df.head(20))
                except Exception as e:
                    st.warning(f"Could not load LST data: {e}")
            
            # Show thumbnail images
            st.subheader("🗺️ Earth Engine Maps")
            
            # Check if URLs are already stored
            if "thumbnail_urls" in params:
                # Display stored URLs
                cols = st.columns(3)
                col_idx = 0
                
                if "lst" in params["thumbnail_urls"]:
                    with cols[col_idx % 3]:
                        st.image(params["thumbnail_urls"]["lst"], caption="Land Surface Temperature (°C)", use_container_width=True)
                    col_idx += 1
                
                if "elevation" in params["thumbnail_urls"]:
                    with cols[col_idx % 3]:
                        st.image(params["thumbnail_urls"]["elevation"], caption="Elevation (m)", use_container_width=True)
                    col_idx += 1
                
                if "land_cover" in params["thumbnail_urls"]:
                    with cols[col_idx % 3]:
                        st.image(params["thumbnail_urls"]["land_cover"], caption="Land Cover", use_container_width=True)
            else:
                # Generate URLs (fallback for old messages)
                with st.spinner("Generating Earth Engine thumbnails..."):
                    try:
                        poi = ee.Geometry.Point(params["lon"], params["lat"])
                        roi = poi.buffer(100000)
                        
                        cols = st.columns(3)
                        col_idx = 0
                        
                        if 'lst_map' in params["visualizations"] or 'all' in params["visualizations"]:
                            with cols[col_idx % 3]:
                                lst_col = ee.ImageCollection('MODIS/061/MOD11A1').select('LST_Day_1km').filterDate(params["start_date"], params["end_date"])
                                lst_img = lst_col.mean().multiply(0.02).add(-273.15)
                                lst_vis = {'min': 0, 'max': 40, 'palette': ['white', 'blue', 'green', 'yellow', 'orange', 'red'], 'dimensions': 512}
                                lst_url = lst_img.getThumbURL({**lst_vis, 'region': roi})
                                st.image(lst_url, caption="Land Surface Temperature (°C)", use_container_width=True)
                            col_idx += 1
                        
                        if 'elevation' in params["visualizations"] or 'all' in params["visualizations"]:
                            with cols[col_idx % 3]:
                                elv_img = ee.Image('USGS/SRTMGL1_003').updateMask(ee.Image('USGS/SRTMGL1_003').gt(0))
                                elv_vis = {'min': 0, 'max': 4000, 'palette': ['006633', 'E5FFCC', '662A00', 'D8D8D8', 'F5F5F5'], 'dimensions': 512}
                                elv_url = elv_img.getThumbURL({**elv_vis, 'region': roi})
                                st.image(elv_url, caption="Elevation (m)", use_container_width=True)
                            col_idx += 1
                        
                        if 'land_cover' in params["visualizations"] or 'all' in params["visualizations"]:
                            with cols[col_idx % 3]:
                                lc_col = ee.ImageCollection('MODIS/061/MCD12Q1').select('LC_Type1')
                                lc_img = lc_col.filterDate(params["start_date"], params["end_date"]).first()
                                if lc_img is None:
                                    lc_img = lc_col.sort('system:time_start', False).first()
                                lc_vis = {
                                    'min': 1, 'max': 17, 'dimensions': 512,
                                    'palette': ['05450a','086a10', '54a708', '78d203', '009900', 'c6b044',
                                               'dcd159', 'dade48', 'fbff13', 'b6ff05', '27ff87', 'c24f44',
                                               'a5a5a5', 'ff6d4c', '69fff8', 'f9ffa4', '1c0dff']
                                }
                                lc_url = lc_img.getThumbURL({**lc_vis, 'region': roi})
                                st.image(lc_url, caption="Land Cover", use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not generate map thumbnails: {e}")

# Chat input
if prompt := st.chat_input("Chat with me about Earth Engine data..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.conversation_history.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process with LLM
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        status_placeholder.info("🤔 Thinking... (Large model may take 30-60 seconds)")
        
        with st.spinner("Processing..."):
            # Call LLM with tools
            try:
                response = call_ollama_with_tools(
                    st.session_state.conversation_history,
                    model=ollama_model,
                    tools=AVAILABLE_TOOLS
                )
                
                if "error" in response:
                    st.error(response["error"])
                    st.stop()
                
                message = response.get("message", {})
                
                # Debug output
                if debug_mode:
                    st.info(f"🔍 **Debug - Full Response:**\n```json\n{json.dumps(response, indent=2)}\n```")
                
                # Check if LLM wants to call a tool
                tool_calls = message.get("tool_calls", [])
                
                # Fallback: If no tool calls but content suggests tool use, try to extract manually
                if not tool_calls and message.get("content"):
                    content = message.get("content", "")
                    # Check if response contains function call in text format
                    if "get_earth_engine_visualization" in content:
                        st.warning("⚠️ Model returned tool call in text format. Trying to parse...")
                        # Try to extract JSON from content
                        json_match = re.search(r'\{[^}]+\}', content)
                        if json_match:
                            try:
                                args = json.loads(json_match.group())
                                if "location" in args:
                                    # Manually construct tool call
                                    tool_calls = [{
                                        "function": {
                                            "name": "get_earth_engine_visualization",
                                            "arguments": args
                                        }
                                    }]
                            except:
                                pass
                
            except Exception as e:
                st.error(f"Error calling LLM: {str(e)}")
                if debug_mode:
                    st.exception(e)
                st.stop()
            
            if tool_calls:
                # LLM decided to use a tool
                for tool_call in tool_calls:
                    function_name = tool_call.get("function", {}).get("name")
                    function_args = tool_call.get("function", {}).get("arguments", {})
                    
                    if debug_mode:
                        st.info(f"🔧 **Tool Call**: {function_name}\n```json\n{json.dumps(function_args, indent=2)}\n```")
                    
                    if function_name == "get_earth_engine_visualization":
                        # Execute the tool
                        result = execute_earth_engine_tool(**function_args)
                        
                        if result.get("success"):
                            # Display results
                            response_text = f"📍 **Location**: {result['location']} ({result['lat']:.4f}, {result['lon']:.4f})\n"
                            response_text += f"📅 **Date Range**: {result['start_date']} to {result['end_date']}\n"
                            response_text += f"🗺️ **Visualizations**: {', '.join(result['visualizations'])}\n\n"
                            
                            st.markdown(response_text)
                            
                            # Store visualization parameters for persistence
                            response_data = {
                                "role": "assistant",
                                "content": response_text,
                                "viz_params": {
                                    "location": result["location"],
                                    "lat": result["lat"],
                                    "lon": result["lon"],
                                    "start_date": result["start_date"],
                                    "end_date": result["end_date"],
                                    "visualizations": result["visualizations"],
                                    "thumbnail_urls": result.get("thumbnail_urls", {})
                                }
                            }
                            
                            # Display visualizations immediately
                            if "lst_plot" in result and result["lst_plot"]:
                                st.subheader("📈 Daytime LST Time Series")
                                st.pyplot(result["lst_plot"])
                                
                                if "lst_data" in result and not result["lst_data"].empty:
                                    with st.expander("View Data Sample"):
                                        st.dataframe(result["lst_data"].head(20))
                            
                            # Show thumbnail images
                            if "thumbnail_urls" in result and result["thumbnail_urls"]:
                                st.subheader("🗺️ Earth Engine Maps")
                                cols = st.columns(3)
                                col_idx = 0
                                
                                if "lst" in result["thumbnail_urls"]:
                                    with cols[col_idx % 3]:
                                        st.image(result["thumbnail_urls"]["lst"], caption="Land Surface Temperature (°C)", use_container_width=True)
                                    col_idx += 1
                                
                                if "elevation" in result["thumbnail_urls"]:
                                    with cols[col_idx % 3]:
                                        st.image(result["thumbnail_urls"]["elevation"], caption="Elevation (m)", use_container_width=True)
                                    col_idx += 1
                                
                                if "land_cover" in result["thumbnail_urls"]:
                                    with cols[col_idx % 3]:
                                        st.image(result["thumbnail_urls"]["land_cover"], caption="Land Cover", use_container_width=True)
                            
                            st.success("✅ Analysis complete!")
                            
                            # Add tool result to conversation
                            st.session_state.conversation_history.append({
                                "role": "tool",
                                "content": json.dumps({"status": "success", "location": result["location"]})
                            })
                            
                            # Get final response from LLM
                            final_response = call_ollama_with_tools(
                                st.session_state.conversation_history,
                                model=ollama_model
                            )
                            
                            final_message = final_response.get("message", {}).get("content", "")
                            if final_message:
                                st.markdown("\n\n" + final_message)
                                response_data["content"] += "\n\n" + final_message
                            
                            st.session_state.messages.append(response_data)
                            st.session_state.conversation_history.append({"role": "assistant", "content": response_data["content"]})
                        else:
                            # Tool execution failed
                            error_msg = result.get("error", "Unknown error")
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": f"❌ {error_msg}"})
                            st.session_state.conversation_history.append({"role": "assistant", "content": error_msg})
            else:
                # Regular conversation response (no tool call)
                response_content = message.get("content", "I'm not sure how to help with that.")
                st.markdown(response_content)
                
                # Fallback: Check if user clearly asked for visualization but tool wasn't called
                keywords = ['map', 'show', 'display', 'visualiz', 'temperature', 'lst', 'elevation', 'land cover']
                if any(keyword in prompt.lower() for keyword in keywords):
                    st.info("💡 **Tip**: It looks like you want a visualization. Let me try to help...")
                    st.markdown("You can also type something like: `Show me [location]` or `Generate map for [city]`")
                
                st.session_state.messages.append({"role": "assistant", "content": response_content})
                st.session_state.conversation_history.append({"role": "assistant", "content": response_content})
