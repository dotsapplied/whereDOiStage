import os
import pandas as pd
import streamlit as st
import numpy as np
from pathlib import Path

# EVE Constants
LY_IN_METERS = 9460528400000000

class EveStagingFinder:
    def __init__(self, sde_data: pd.DataFrame):
        """
        Initialize with flattened SDE data.
        """
        self.systems = sde_data.copy()
        self.systems['solarSystemName'] = self.systems['solarSystemName'].astype(str)
        self.systems['name_lower'] = self.systems['solarSystemName'].str.lower()

    def find_best_staging(self, targets: list, jump_range: float, ignore_highsec: bool):
        potential_staging = self.systems.copy()
        if ignore_highsec:
            # 0.45 is the internal cutoff for 0.5 (highsec)
            potential_staging = potential_staging[potential_staging['security'] < 0.45].copy()

        search_targets = [t.lower() for t in targets]
        target_data = self.systems[self.systems['name_lower'].isin(search_targets)]
        
        if target_data.empty:
            return []

        # Use NumPy for high-performance distance matrix calculation
        staging_coords = potential_staging[['x', 'y', 'z']].values
        target_coords = target_data[['x', 'y', 'z']].values
        target_names = target_data['solarSystemName'].values

        results = []
        for i, st_coord in enumerate(staging_coords):
            # Calculate distance from this potential staging to all targets
            diffs = target_coords - st_coord
            dists_meters = np.sqrt(np.sum(diffs**2, axis=1))
            dists_ly = dists_meters / LY_IN_METERS
            
            mask = dists_ly <= jump_range
            reachable_count = np.sum(mask)

            if reachable_count > 0:
                staging_sys = potential_staging.iloc[i]
                dist_details = [f"{target_names[j]} - {round(dists_ly[j], 2)} LY" for j in range(len(mask)) if mask[j]]
                results.append({
                    "Staging System": staging_sys['solarSystemName'],
                    "Security": round(staging_sys['security'], 1),
                    "NPC Station": staging_sys['hasNPCStation'],
                    "Targets Covered": reachable_count,
                    "Distances": ", ".join(dist_details)
                })

        # Sort by most targets covered, then by security (lower is usually better for nullsec staging)
        return sorted(results, key=lambda x: (x['Targets Covered'], -x['Security']), reverse=True)

# --- Streamlit UI ---

@st.cache_data
def load_sde_data():
    # Determine the folder where this script is located
    base_path = Path(__file__).parent
    jsonl_path = base_path / "mapSolarSystems.jsonl"
    npc_path = base_path / "npcStations.jsonl"
    
    if jsonl_path.exists():
        try:
            # Load JSONL (JSON Lines) format
            df = pd.read_json(jsonl_path, lines=True)
            
            # Flatten the nested structure found in the SDE JSONL
            df['solarSystemName'] = df['name'].apply(lambda x: x.get('en'))
            df['x'] = df['position'].apply(lambda x: x.get('x'))
            df['y'] = df['position'].apply(lambda x: x.get('y'))
            df['z'] = df['position'].apply(lambda x: x.get('z'))
            df['security'] = df['securityStatus']
            
            # Cross-reference with NPC stations
            if npc_path.exists():
                stations_df = pd.read_json(npc_path, lines=True)
                station_system_ids = set(stations_df['solarSystemID'].unique())
                df['hasNPCStation'] = df['_key'].isin(station_system_ids)
            else:
                df['hasNPCStation'] = False
            
            return df[['solarSystemName', 'x', 'y', 'z', 'security', 'hasNPCStation']]
        except Exception as e:
            st.error(f"Error parsing SDE file: {e}")
            return None
    return None

def main():
    st.set_page_config(page_title="whereDOiStage", page_icon="🛰️", layout="wide", initial_sidebar_state="expanded")
    
    # Custom CSS for a modern "EVE" look
    st.markdown("""
        <style>
        .main { background-color: #0e1117; }
        .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
        [data-testid="stSidebar"] { background-color: #161b22; }
        .stDataFrame { border: 1px solid #30363d; border-radius: 10px; }
        </style>
    """, unsafe_allow_html=True)

    st.title("🛰️ whereDOiStage")
    st.caption("coco's ai slop")

    st.sidebar.header("Jump Configuration")
    range_option = st.sidebar.radio(
        "Select Hull Type (Jump Range):",
        ("6 LY (Super/Titan)", "7 LY (Capitals)", "8 LY (Black Ops)", "10 LY (JF/Rorqual)", "Custom")
    )
    jump_range = float(range_option.split()[0]) if range_option != "Custom" else st.sidebar.number_input("LY:", value=5.0)
    
    ignore_hs = st.sidebar.checkbox("Ignore Highsec Systems (>= 0.5)", value=True)

    target_input = st.text_input("Enter Target Systems (comma separated):", placeholder="NOL-M9, 1DQ1-A, Jita")
    target_list = [x.strip() for x in target_input.split(",") if x.strip()]

    sde = load_sde_data()
    if sde is None:
        st.error(f"Could not find 'mapSolarSystems.jsonl' in {Path(__file__).parent}")
        st.stop()

    finder = EveStagingFinder(sde)

    if target_list:
        found_mask = sde['solarSystemName'].str.lower().isin([t.lower() for t in target_list])
        found_names = sde[found_mask]['solarSystemName'].tolist()
        missing = [t for t in target_list if t.lower() not in [f.lower() for f in found_names]]
        
        if missing:
            st.warning(f"Systems not found in SDE: {', '.join(missing)}")

        results = finder.find_best_staging(target_list, jump_range, ignore_hs)
        
        if results:
            # Dashboard Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Targets Found", len(found_names))
            m2.metric("Max Coverage", f"{results[0]['Targets Covered']} Systems")
            m3.metric("Jump Range", f"{jump_range} LY")

            st.divider()
            st.subheader("📍 Recommended Staging Locations")
            
            df_results = pd.DataFrame(results)
            st.dataframe(
                df_results[['Staging System', 'Security', 'NPC Station', 'Targets Covered', 'Distances']],
                use_container_width=True,
                height=600,
                column_config={
                    "Security": st.column_config.NumberColumn("Sec", format="%.1f"),
                    "NPC Station": st.column_config.CheckboxColumn("NPC Station"),
                    "Targets Covered": st.column_config.ProgressColumn(
                        "Coverage", 
                        min_value=0, 
                        max_value=len(found_names),
                        format="%d systems"
                    ),
                    "Distances": st.column_config.TextColumn("Target Distances")
                }
            )
        elif found_names:
            st.warning("No staging systems found within that range for the given targets.")
    else:
        st.info("💡 Enter system names above (e.g., 1DQ1-A, NOL-M9) to find the best central staging point.")

if __name__ == "__main__":
    main()