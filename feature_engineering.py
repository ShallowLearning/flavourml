import numpy as np

def triangle_function(x1,x2,x3):
    return (x1**2 - x2**2 - x3**2)**2 - 4 * x1**2 * x2**2

def derive_features(df):
    # eta diffs
    df['JOSH_delta_eta_p0p1'] = np.abs(df.p0_eta - df.p1_eta)
    df['JOSH_delta_eta_p0p2'] = np.abs(df.p0_eta - df.p2_eta)
    df['JOSH_delta_eta_p1p2'] = np.abs(df.p1_eta - df.p2_eta)

    # eta ratios
    df['JOSH_ratio_eta_p0p1'] = df.p0_eta / np.clip(df.p1_eta, 1e-15, np.max(df.p1_eta))
    df['JOSH_ratio_eta_p0p2'] = df.p0_eta / np.clip(df.p2_eta, 1e-15, np.max(df.p1_eta))
    df['JOSH_ratio_eta_p1p2'] = df.p1_eta / np.clip(df.p2_eta, 1e-15, np.max(df.p1_eta))
    
    # momentum products
    df['JOSH_pt_p0p1'] = df.p0_pt * df.p1_pt
    df['JOSH_pt_p0p2'] = df.p0_pt * df.p2_pt
    df['JOSH_pt_p1p2'] = df.p1_pt * df.p2_pt

    # momentum asymmetries
    df['JOSH_asym_pt0pt1'] = np.abs(df.p0_pt - df.p1_pt) / (df.p0_pt + df.p1_pt)
    df['JOSH_asym_pt0pt2'] = np.abs(df.p0_pt - df.p2_pt) / (df.p0_pt + df.p2_pt)
    df['JOSH_asym_pt1pt2'] = np.abs(df.p1_pt - df.p2_pt) / (df.p1_pt + df.p2_pt)

    # longitudinal momentum
    df['JOSH_p0_pz'] = df.p0_pt * np.sinh(df.p0_eta)
    df['JOSH_p1_pz'] = df.p1_pt * np.sinh(df.p1_eta)
    df['JOSH_p2_pz'] = df.p2_pt * np.sinh(df.p2_eta)
    
    # transverse mass
    df['JOSH_ml_p0p1'] = np.sqrt(2 * df.p0_pt * df.p1_pt * np.cosh(df.p0_eta - df.p1_eta))
    df['JOSH_ml_p0p2'] = np.sqrt(2 * df.p0_pt * df.p2_pt * np.cosh(df.p0_eta - df.p2_eta))
    df['JOSH_ml_p1p2'] = np.sqrt(2 * df.p1_pt * df.p2_pt * np.cosh(df.p1_eta - df.p2_eta))

    # decay momentum ratios
    df['JOSH_tau_p0_pt_ratio'] = df['p0_pt'] / df['pt']
    df['JOSH_tau_p1_pt_ratio'] = df['p1_pt'] / df['pt']
    df['JOSH_tau_p2_pt_ratio'] = df['p2_pt'] / df['pt']
    
    # IP variables
    df['JOSH_IP_norm'] = df['IP'] / np.sqrt(df['IPSig'])
    df['JOSH_IP_dira'] = df['IP'] * df['dira']

    # chi2 variables

    # flight distance
    df['JOSH_FlightDistance_norm'] = df['FlightDistance'] / np.sqrt(df['FlightDistanceError'])
    df['JOSH_LifeTime_FlightDistance'] = df['JOSH_FlightDistance_norm'] * df['LifeTime']
