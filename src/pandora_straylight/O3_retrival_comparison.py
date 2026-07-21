import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# === Configuration ===
file_p2 = r"data\O3_retrieval\Pandora2s1_GreenbeltMD_L2_rout2p1-8_online_apr_14_30.txt"
# Pandora 63
file = r"data\O3_retrieval\Pandora63s1_GreenbeltMD_L2_rout2p1-8_apr_14_30.txt"
file_corrected = r"data\O3_retrieval\Pandora63s1_GreenbeltMD_L2_rout2p1-8_apr_14_30_ib20.txt"
columns_to_plot = [9, 19, 20, 39]

def load_l2_data(file_path):
    with open(file_path, 'r', encoding='latin1') as f:
        lines = f.readlines()

    # Extract title
    plot_title = "L2 Data"
    for line in lines:
        if line.startswith("Data product status:"):
            plot_title = line.strip().split("Data product status:")[-1].strip()

    # Find data start
    data_start = next(i for i, line in enumerate(lines) if line.strip().startswith("----")) + 1

    # Parse data
    records = []
    for line in lines[data_start:]:
        if line.strip() and line[0].isdigit():
            parts = line.strip().split()
            if len(parts) >= max(columns_to_plot):
                records.append(parts)

    df = pd.DataFrame(records)
    df[0] = pd.to_datetime(df[0].str[:15], format="%Y%m%dT%H%M%S")
    for col in columns_to_plot:
        df[col - 1] = pd.to_numeric(df[col - 1], errors='coerce')

    df['date_only'] = df[0].dt.date
    midpoints = df.groupby('date_only')[0].agg(lambda x: x.min() + (x.max() - x.min()) / 2)

    return df, midpoints, plot_title

# Load L2 files
df_p2, midpoints_p2, title_p2 = load_l2_data(file_p2)

df0, midpoints0, title0 = load_l2_data(file)
df1, midpoints1, title1 = load_l2_data(file_corrected)

# Combine x-ticks from both datasets
combined_midpoints = pd.Series(pd.concat([midpoints_p2])).sort_values().drop_duplicates()

# === Plot ===
fig, ax = plt.subplots(figsize=(12, 6))

# # Reference Instrument: Pandora 2
ax.plot(
    df_p2[0],
    df_p2[39 - 1] * 1000,
    marker='o',
    linestyle='None',
    markersize=3,
    alpha=0.7,
    color ='black',
    label='Pandora 2'
)


# File 0: Pandora 63 Before corrected
ax.plot(
    df0[0],
    df0[39 - 1] * 1000,
    marker='o',
    linestyle='None',
    markersize=3,
    alpha=0.7,
    color ='blue',
    label='Pandora 63'
)

# File 1: corrected
ax.plot(
    df1[0],
    df1[39 - 1] * 1000,
    marker='o',
    linestyle='None',
    markersize=3,
    alpha=0.7,
    color='green',
    label='Pandora 63, corrected ib=20'
)

# Format
ax.set_ylabel("O₃ column (mmol/m²)")
ax.set_ylim(110, 200)
ax.set_title(f"{title_p2} comparison")
ax.set_xticks(combined_midpoints.values)
ax.set_xticklabels([d.strftime("%m-%d") for d in combined_midpoints.index], rotation=90)
ax.tick_params(labelbottom=True)
ax.legend()
plt.tight_layout()
plt.show()
