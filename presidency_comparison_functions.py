# Python 3.13 - 29 Sept 2025
# presidency_comparison_functions.py

import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd

def visual(df, kind=None, double=False): # double = True implies double standardization of tariff data (sector*year)
    if kind == 'heatmap':
        # Ensure NAICS is the index and years are columns
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            df,
            cmap="RdBu_r",  # red = above average, blue = below
            center=0,  # mean = 0 because of standardization
            cbar_kws={'label': legend}
        )

        plt.title(main, fontsize=14)
        plt.xlabel(x_lab)
        plt.ylabel(y_lab)
        plt.tight_layout()
        plt.savefig("heatmap_tariffs.png", dpi=300)
        plt.show()

    if kind == 'faceted_lg':
        if double == False:  # If double is set == True, the input dataset is already melted.
            # Convert wide format into long format for seaborn
            df = df.melt(
                id_vars='SECTOR',
                var_name='Year',
                value_name='Tariff'
            )
            df['Year'] = df['Year'].astype(np.int64)
        else:
            df['Year'] = df['Year'].astype(np.int64)

        # FacetGrid: one subplot per NAICS3 sector
        df['SECTOR'] = df['SECTOR'].apply(
            lambda x: "\n".join(textwrap.wrap(x, 20))
        )  # Avoiding overlapping between plots in the same row

        g = sns.FacetGrid(
            df,
            col="SECTOR",
            col_wrap=6,  # number of columns in grid
            sharey=True,  # each sector has its own y-scale
            height=2.2
        )

        g.map(sns.lineplot, "Year", "Tariff", marker="o")
        g.set_titles("{col_name}", fontsize=10)
        g.set_axis_labels("Year", "Standardized Tariff Level")

        plt.subplots_adjust(top=0.92)
        g.fig.suptitle('Tariffs evolution over the years per NAICS (note: double standardization is applied)', fontweight="bold", fontsize=11)
        # g.savefig("faceted_lineplots.png", dpi=300)

        for ax in g.axes.flatten():
            ax.axvline(x=2021, color="orange", linestyle="--", linewidth=1)
            ax.axvline(x=2022, color="green", linestyle="--", linewidth=1)
        plt.savefig('faceted_lineplots.png', dpi=300)
        plt.show()


# Make sure the data frame has the 'SECTOR' column.
def standardizer(data_frame, st_type=None):
    # Standardize by year (column) to show the evolution in the sector's share of total duties paid over the years
    # Statistical note: high values in absolute value are far away from the mean of duties paid per sector in that specific year
    if st_type == 'year':
        data_frame = data_frame.groupby('SECTOR')[
            ['2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024']].sum()
        st_year = data_frame.copy()
        for year in ['2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024']:
            mean = st_year[year].mean()
            sd = st_year[year].std()
            st_year[year] = st_year[year].apply(
                lambda x: (x - mean) / sd
            )
        return st_year

    # Standardize by sector (row) to show the trend in total amount of duties paid by each sector over the years
    # Statistical note: high values in absolute value are far away from the mean of duties paid by that sector over the years
    if st_type == 'sector':
        data_frame = data_frame.groupby('SECTOR')[
            ['2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024']].sum()
        st_sector = data_frame.copy()
        for sector in st_sector.index:
            mean = st_sector.loc[sector].mean()
            sd = st_sector.loc[sector].std()
            st_sector.loc[sector] = st_sector.loc[sector].apply(
                lambda x: (x - mean) / sd
            )
        return st_sector

    # Standardize by sector AND year.
    if st_type == 'double':  # I need the dataset with NAICS codes.
        df_long = data_frame.copy().melt(
            id_vars=['SECTOR', 'naics4', 'NAICS', 'NAICS2', 'HTS Number'],
            var_name='Year',
            value_name='Tariff'
        )
        df_long['Year'] = df_long['Year'].astype(np.int64)
        df_long['Tariff'] = df_long['Tariff'].astype(str).apply(
            lambda x: x.replace(',', '')
        ).astype(np.int64)

        # Compute two-way demeaning residuals: resid_ij = X_ij - sector_mean_i - year_mean_j + overall_mean
        df_long = (
            df_long
            .groupby(['Year', 'SECTOR'], as_index=False)
            .agg({
                'Tariff': 'sum',
                'NAICS': 'first',
                'NAICS2': 'first',
                'naics4': 'first'
            })
        )

        df_long['sector_mean'] = df_long.groupby('SECTOR')['Tariff'].transform('mean')  # mean of COLUMNS
        df_long['year_mean'] = df_long.groupby('Year')['Tariff'].transform('mean')  # mean of ROWS
        overall_mean = df_long['Tariff'].mean()

        df_long['resid_two_way'] = df_long['Tariff'] - df_long['sector_mean'] - df_long['year_mean'] + overall_mean

        # Sanity: the sector and year means (after removing the other) should sum to zero across groups e.g. sector-level mean of resid_two_way should be ~0:
        # df_long.groupby('SECTOR')['resid_two_way'].mean()

        # Standardize the residuals (z-score) using StandardScaler
        scaler = StandardScaler(with_mean=True,
                                with_std=True,
                                copy=True)

        # transform expects 2D array; keep results aligned to df_long
        df_long['resid_two_way_z'] = scaler.fit_transform(df_long[['resid_two_way']])[:, 0]
        df_long = df_long.drop('resid_two_way', axis=1)
        return df_long


# CLASSIFICATION FUNCTION #

def classify_sector(naics, n=None): # n=3 or n=4 are possible (NAICS3 vs NAICS4)
    if pd.isna(naics):
        return 'Unknown'
    naics = str(naics)
    if n == 3:
        naics = naics[:3]
    
        tech_hardware = ['334']  # Computer and Electronic Product Manufacturing (semiconductors, computer hardware, telecom equipment)
        # TRADITIONAL MANUFACTURING (including Tesla and automotive)
        
        traditional_manufacturing = [
            '331',  # Primary Metal Manufacturing
            '332',  # Fabricated Metal Product Manufacturing  
            '333',  # Machinery Manufacturing
            '335',  # Electrical Equipment, Appliance, and Component Manufacturing
            '336',  # Transportation Equipment Manufacturing (TESLA)
            '337'   # Furniture and Related Product Manufacturing
        ]
        
        high_protection = [
            '313', '314', '315', '316',  # Textiles & Apparel
            '311', '312'  # Food Manufacturing
        ]
        other_manufacturing = ['321', '322', '323', '324', '325', '326', '327', '339']
        primary_sectors = ['111', '112', '113', '114', '115']
        
        if naics in tech_hardware:
            return 'Tech Hardware'
        elif naics in traditional_manufacturing:
            return 'Traditional Manufacturing'
        elif naics in high_protection:
            return 'High Protection Sectors'
        elif naics in other_manufacturing:
            return 'Other Manufacturing'
        elif naics in primary_sectors:
            return 'Agriculture & Primary'
        elif naics.startswith('2'):
            return 'Mining & Construction'
        else:
            return 'Other'

    elif n == 4:
        # === DIGITAL SERVICES PROXIES (via goods che usano questi servizi) ===
        digital_intensive = {
            # Computer & Electronics (334X)
            '3341': 'Computer Manufacturing', # Amazon vende PC, laptops
            '3342': 'Communications Equipment', # Phones, routers via e-commerce
            '3343': 'Audio/Video Equipment', # Consumer electronics via Amazon
            '3344': 'Semiconductors', # Components (meno consumer-facing)
            '3345': 'Instruments', # Measurement instruments
            '3346': 'Magnetic Media', # Disks, storage
            '3231': 'Printing', # Printing & Publishing (323X)                         
            '3151': 'Apparel Knitting', # Apparel (315X) - heavy e-commerce
            '3152': 'Cut & Sew Apparel',
            '3159': 'Apparel Accessories',
            '3371': 'Household Furniture', # Furniture (337X) - e-commerce growing
            '3372': 'Office Furniture',
        }
        # === LOGISTICS-INTENSIVE GOODS (Amazon's delivery network) ===
        logistics_intensive = {
            # Small consumer goods facilmente spedibili
            '3399': 'Other Miscellaneous Mfg',          # Small consumer products
            '3169': 'Other Leather Products',           # Shoes, bags
            '3262': 'Rubber Products',                  # Consumer rubber goods
        }
        # === TRADITIONAL B2B (LOW Big Tech relevance) ===
        traditional_b2b = {
            # Automotive (336X) - granularità per separare Tesla
            '3361': 'Motor Vehicles',                   # eg Tesla
            '3362': 'Motor Vehicle Bodies',             # B2B
            '3363': 'Motor Vehicle Parts',              # B2B supply chain
            '3364': 'Aerospace',                        # B2B, defense
            '3365': 'Railroad',                         # B2B
            '3366': 'Ship & Boat Building',             # B2B
            '3369': 'Other Transport Equipment',        # B2B
            
            # Heavy manufacturing
            '3311': 'Iron & Steel',                     # B2B commodities
            '3312': 'Steel Products',
            '3313': 'Alumina & Aluminum',
            '3315': 'Foundries',
            
            '3321': 'Forging & Stamping',               # B2B inputs
            '3322': 'Cutlery & Handtools',
            '3323': 'Architectural Metal',
            '3324': 'Boilers & Tanks',
            '3325': 'Hardware',
            '3326': 'Springs & Wire',
            '3327': 'Machine Shops',
            '3328': 'Coatings & Engravings',
            '3329': 'Other Fabricated Metal',
            
            '3331': 'Agriculture Machinery',            # B2B
            '3332': 'Industrial Machinery',
            '3333': 'Commercial Machinery',
            '3334': 'HVAC',
            '3335': 'Metalworking Machinery',
            '3336': 'Engines & Turbines',
            '3339': 'Other Machinery',
        }
        
        # Classificazione
        if naics in digital_intensive:
            return f"Digital-Intensive: {digital_intensive[naics]}"
        elif naics in logistics_intensive:
            return f"Logistics-Intensive: {logistics_intensive[naics]}"
        elif naics in traditional_b2b:
            return f"Traditional B2B: {traditional_b2b[naics]}"
        else:
            return "Other Sectors"

def prepare_data(data_frame, groups=None):
    # groups keyword arg can take three values:
    
    # If 'sector': data is grouped based on sector group
    # If 'naics3': data is grouped based on NAICS3
    # If 'naics4': data is grouped based on NAICS4
    
    """Prepara i dati per l'analisi Trump vs Biden"""
    
    data_frame['Year'] = data_frame['Year'].astype(int)
    data_frame['Tariff'] = pd.to_numeric(
        data_frame['Tariff']
            .astype(str)
            .str.replace(',', ''),
        errors='coerce'
    )
    if groups == 'sector':
        data_frame['SECTOR_GROUP'] = data_frame['NAICS'].apply(classify_sector, n=3)
        # AGGREGATION
        df_agg = (
            data_frame
            .groupby(['Year', 'SECTOR_GROUP'], as_index=False)
            .agg({'Tariff': 'sum'})
        )
        # Sector groups
        df_agg['is_tech_hardware'] = (df_agg['SECTOR_GROUP'] == 'Tech Hardware').astype(int)
        df_agg['is_traditional'] = (df_agg['SECTOR_GROUP'] == 'Traditional Manufacturing').astype(int)
        df_agg['is_high_protection'] = (df_agg['SECTOR_GROUP'] == 'High Protection Sectors').astype(int)

    elif groups == 'naics3':
        df_agg = (data_frame.groupby(['Year', 'NAICS'], as_index=False).agg({'Tariff':'sum', 'resid_two_way_z':'first'}))
        df_agg['SECTOR_GROUP'] = df_agg['NAICS'].apply(classify_sector, n=3)
        # Sector groups
        df_agg['is_tech_hardware'] = (df_agg['SECTOR_GROUP'] == 'Tech Hardware').astype(int)
        df_agg['is_traditional'] = (df_agg['SECTOR_GROUP'] == 'Traditional Manufacturing').astype(int)
        df_agg['is_high_protection'] = (df_agg['SECTOR_GROUP'] == 'High Protection Sectors').astype(int)
        
    elif groups == 'naics4':
        df_agg = (data_frame.groupby(['Year', 'naics4'], as_index=False).agg({'Tariff':'sum', 'resid_two_way_z':'first'}))
        df_agg['Granular_Category'] = df_agg['naics4'].apply(classify_sector, n=4)
        # Crea dummy per categorie aggregate
        df_agg['is_digital_intensive'] = df_agg['Granular_Category'].str.contains('Digital-Intensive', na=False).astype(int)
        df_agg['is_logistics_intensive'] = df_agg['Granular_Category'].str.contains('Logistics-Intensive', na=False).astype(int)
        df_agg['is_traditional_b2b'] = df_agg['Granular_Category'].str.contains('Traditional B2B', na=False).astype(int)
    
    # Presidency
    df_agg['Presidency'] = df_agg['Year'].apply(lambda y: 'Trump' if y <= 2020 else 'Biden')
    df_agg['PresidencyNum'] = df_agg['Presidency'].map({'Trump':0, 'Biden':1})
    
    return df_agg

# Useful for EDA. Not so relevant otherwise.
def analyze_sector_groups(data_by_year):
    """Analizza la distribuzione dei settori e delle tariffe per gruppo"""
    data_by_year['SECTOR_GROUP'] = data_by_year['NAICS'].apply(classify_sector)
    print("=== DISTRIBUZIONE DEI SETTORI PER GRUPPO ===")
    group_counts = data_by_year.groupby('SECTOR_GROUP')['HTS Number'].nunique().sort_values(ascending=False)
    print(group_counts)
    print()
    
    # Tariffe totali per gruppo nel 2024 (anno più recente) - usa copia temporanea
    print("=== TARIFFE TOTALI 2024 PER GRUPPO (in $) ===")
    df_temp = data_by_year.copy()
    df_temp['2024_clean'] = pd.to_numeric(df_temp['2024'].astype(str).str.replace(',', ''), errors='coerce')
    tariffs_2024 = df_temp.groupby('SECTOR_GROUP')['2024_clean'].sum().sort_values(ascending=False)
    for group, amount in tariffs_2024.items():
        print(f"{group}: ${amount:,.0f}")
    print()
    
    return group_counts, tariffs_2024

# proportions() is called to build the stackplot and it makes sure that tariffs across sectors are expressed as a %.
def proportions(df, group_col='SECTOR_GROUP', year_col='Year', value_col='Tariff'):
    df_prop = df.copy()
    # Calcola totale per anno
    year_totals = df_prop.groupby(year_col)[value_col].transform('sum')
    # Calcola proporzioni
    df_prop['Tariff_Proportion'] = df_prop[value_col] / year_totals
    
    return df_prop