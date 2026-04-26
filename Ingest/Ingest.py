"""
COT Prediction Ingest
Reads Rollex (daily) + COT (weekly) from ICEBREAKER source folders.
Stores both spec net variants (with / without index) so the dashboard
can toggle between them and fit its own regression dynamically.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

BASE       = Path(r"C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\ICEBREAKER")
COT_DIR    = BASE / "COT" / "Database"
ROLLEX_DIR = BASE / "Rollex" / "Database"
OUT_DIR    = Path(__file__).resolve().parent.parent / "Database"

COMMODITY_CONFIG = {
    'KC':  {'name': 'Arabica',      'report': 'CIT',   'spec_full': 'Non Com + Non Rep + Index', 'spec_no_idx': 'Non Com + Non Rep'},
    'CC':  {'name': 'NYC Cocoa',    'report': 'CIT',   'spec_full': 'Non Com + Non Rep + Index', 'spec_no_idx': 'Non Com + Non Rep'},
    'SB':  {'name': 'Sugar',        'report': 'CIT',   'spec_full': 'Non Com + Non Rep + Index', 'spec_no_idx': 'Non Com + Non Rep'},
    'CT':  {'name': 'Cotton',       'report': 'CIT',   'spec_full': 'Non Com + Non Rep + Index', 'spec_no_idx': 'Non Com + Non Rep'},
    'RC':  {'name': 'Robusta',      'report': 'Disagg','spec_full': 'MM + Others + Non Rep',     'spec_no_idx': 'MM + Non Rep'},
    'LCC': {'name': 'London Cocoa', 'report': 'Disagg','spec_full': 'MM + Others + Non Rep',     'spec_no_idx': 'MM + Non Rep'},
}


def build_weekly(sym, cfg, cit, disagg, rollex_px):
    """Align COT Tuesdays with Rollex px; compute both spec net variants."""
    if cfg['report'] == 'CIT':
        raw = cit[cit['Commodity'] == sym].copy()
        non_com = raw['Spec Long']    - raw['Spec Short']
        index   = raw['Index Long']   - raw['Index Short']
        non_rep = raw['Non Rep Long'] - raw['Non Rep Short']
        full     = (non_com + index + non_rep) / 1000
        no_idx   = (non_com + non_rep)          / 1000
    else:
        raw = disagg[disagg['Commodity'] == sym].copy()
        mm      = raw['MM Long']      - raw['MM Short']
        others  = raw['Other Long']   - raw['Other Short']
        non_rep = raw['Non Rep Long'] - raw['Non Rep Short']
        full     = (mm + others + non_rep) / 1000
        no_idx   = (mm + non_rep)           / 1000

    cot = pd.DataFrame({
        'Date':          raw['Date'].values,
        'Px':            raw['Px'].values,
        'Spec_Net_Full': full.values,
        'Spec_Net_NoIdx':no_idx.values,
    }).sort_values('Date').reset_index(drop=True)

    cot['Rollex_Px'] = cot['Date'].apply(lambda d: rollex_px.asof(d))

    return cot


def zscore_filter(df, col):
    """Return mask of rows within 10 std — removes data-error outliers."""
    z = np.abs(stats.zscore(df[col].dropna()))
    clean_idx = df[col].dropna().index[z < 10]
    return df.index.isin(clean_idx)


def main():
    cit    = pd.read_parquet(COT_DIR / "cot_cit.parquet")
    disagg = pd.read_parquet(COT_DIR / "cot_disagg.parquet")

    history_parts = []
    sim_rows      = []

    for sym, cfg in COMMODITY_CONFIG.items():
        rollex_px = pd.read_parquet(ROLLEX_DIR / f"rollex_{sym}.parquet")['rollex_px'].sort_index()

        weekly = build_weekly(sym, cfg, cit, disagg, rollex_px)

        # Add pct changes
        weekly['Rollex_Pct']           = weekly['Rollex_Px'].pct_change()
        weekly['Spec_Net_Full_Chg']    = weekly['Spec_Net_Full'].diff()
        weekly['Spec_Net_NoIdx_Chg']   = weekly['Spec_Net_NoIdx'].diff()

        weekly['Commodity']    = sym
        weekly['Name']         = cfg['name']
        weekly['Report_Type']  = cfg['report']
        weekly['Spec_Full_Label']  = cfg['spec_full']
        weekly['Spec_NoIdx_Label'] = cfg['spec_no_idx']
        weekly['Is_Simulation']    = False

        # Clean for regression diagnostics (print only)
        reg_ready = weekly.dropna(subset=['Rollex_Pct', 'Spec_Net_Full_Chg'])
        mask = zscore_filter(reg_ready, 'Spec_Net_Full_Chg')
        clean = reg_ready[mask]
        if len(clean) > 5:
            s, i, r, _, _ = stats.linregress(clean['Rollex_Pct'], clean['Spec_Net_Full_Chg'])
            r2 = r ** 2
        else:
            r2 = np.nan

        history_parts.append(weekly)

        # --- Simulation row (partial current week) ---
        last_cot_date    = weekly['Date'].max()
        last_full        = weekly.loc[weekly['Date'] == last_cot_date, 'Spec_Net_Full'].iloc[0]
        last_no_idx      = weekly.loc[weekly['Date'] == last_cot_date, 'Spec_Net_NoIdx'].iloc[0]
        baseline_px      = rollex_px.asof(last_cot_date)
        latest_px        = float(rollex_px.iloc[-1])
        latest_date      = rollex_px.index[-1]
        partial_pct      = (latest_px - baseline_px) / baseline_px

        sim_rows.append({
            'Date':              last_cot_date + pd.Timedelta(days=7),
            'Px':                np.nan,
            'Spec_Net_Full':     np.nan,
            'Spec_Net_NoIdx':    np.nan,
            'Rollex_Px':         latest_px,
            'Rollex_Pct':        partial_pct,
            'Spec_Net_Full_Chg': np.nan,
            'Spec_Net_NoIdx_Chg':np.nan,
            'Commodity':         sym,
            'Name':              cfg['name'],
            'Report_Type':       cfg['report'],
            'Spec_Full_Label':   cfg['spec_full'],
            'Spec_NoIdx_Label':  cfg['spec_no_idx'],
            'Is_Simulation':     True,
            'Last_COT_Date':     last_cot_date,
            'Last_Spec_Full':    last_full,
            'Last_Spec_NoIdx':   last_no_idx,
            'Sim_As_Of':         latest_date,
        })

        print(f"  {cfg['name']:15s}  R2(full)={r2:.3f}  last_cot={last_cot_date.date()}  partial={partial_pct:+.2%}")

    history_df = pd.concat(history_parts, ignore_index=True)
    sim_df     = pd.DataFrame(sim_rows)
    out        = pd.concat([history_df, sim_df], ignore_index=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_DIR / "cot_prediction.parquet", index=False)
    print(f"\nSaved {len(out)} rows -> {OUT_DIR / 'cot_prediction.parquet'}")


if __name__ == '__main__':
    main()
