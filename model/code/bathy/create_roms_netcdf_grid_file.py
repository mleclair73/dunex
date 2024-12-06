# Inspired by https://code.usgs.gov/coawstmodel/COAWST/-/blob/daf8f885a3292ca07c1f168fbe11814363387ed9/Tools/mfiles/mtools/create_roms_netcdf_grid_file.m

import numpy as np
import xarray as xr
import pandas as pd

def create_roms_netcdf_grid_file(the_roms_file, LP, MP):
    """
    Creates a NetCDF file with ROMS grid variables.

    Args:
        the_roms_file (str): The path to the ROMS NetCDF output file.
        LP (int): The number of grid points in the xi (left-right) direction.
        MP (int): The number of grid points in the eta (up-down) direction.
    """

    # Open the ROMS file
    ds = xr.Dataset()
    print(" ## Defining Global Attributes...")

    # Global attributes
    ds.attrs["type"] = "ROMS GRID file"
    ds.attrs["gridid"] = "theGridTitle"
    ds.attrs["history"] = f"Created by {__name__}, on {str(pd.Timestamp.now())}"
    ds.attrs["title"] = "ROMS Application"

    # Dimensions
    L = LP - 1  # The psi dimension
    M = MP - 1  # The psi dimension

    print(" ## Defining Dimensions...")

    ds = ds.assign_coords(
        xi_psi=np.arange(L),
        xi_rho=np.arange(LP),
        xi_u=np.arange(L),
        xi_v=np.arange(LP),
        eta_psi=np.arange(M),
        eta_rho=np.arange(MP),
        eta_u=np.arange(MP),
        eta_v=np.arange(M),
        one=np.array([1]),
        two=np.array([1, 2]),
        bath=np.array([1]),
    )

    print(" ## Defining Variables and Attributes...")

    # Variables and attributes
    ds["xl"] = xr.DataArray(
        data=np.nan, 
        dims=("one")
        coords={"one": ds.one},
        attrs={"long_name": "domain length in the XI-direction", "units": "meter"}
    )
    ds["el"] = xr.DataArray(
        data=np.nan, 
        dims=("one")
        coords={"one": ds.one},
        attrs={"long_name": "domain length in the ETA-direction", "units": "meter"}
    )
    ds["JPRJ"] = xr.DataArray(
        data=np.array([b"ME", b"ST"]),
        coords={"two": ds.two},
        attrs={
            "long_name": "Map projection type",
            "option_ME": "Mercator",
            "option_ST": "Stereographic",
            "option_LC": "Lambert conformal conic",
        },
    )
    
    ds["spherical"] = xr.DataArray(
        data=np.array([b"T"]),
        dims=("one")
        coords={"one": ds.one},
        attrs={"long_name": "Grid type logical switch", "option_T": "spherical", "option_F": "Cartesian"},
    )
    ds["depthmin"] = xr.DataArray(
        data=np.nan, 
        dims=("one")    
        coords={"one": ds.one},
        attrs={"long_name": "domain length in the XI-direction", "units": "meter"}
    )
    ds["depthmax"] = xr.DataArray(
        data=np.nan,
        dims=("one")
        coords={"one": ds.one},
        attrs={"long_name": "Deep bathymetry clipping depth", "units": "meter"}
    )
    ds["hraw"] = xr.DataArray(
        data=np.nan,
        dims=("xi_rho", "eta_rho", "bath"),
        coords={"xi_rho": ds.xi_rho, "eta_rho": ds.eta_rho, "bath": ds.bath},
        attrs={"long_name": "Working bathymetry at RHO-points", "units": "meter", "field": "bath, scalar"},
    )
    ds["h"] = xr.DataArray(
        data=np.nan,
        dims=("xi_rho", "eta_rho"),
        coords={"xi_rho": ds.xi_rho, "eta_rho": ds.eta_rho},
        attrs={"long_name": "Final bathymetry at RHO-points", "units": "meter", "field": "bath, scalar"},
    )
    ds["f"] = xr.DataArray(
        data=np.nan,
        dims=("xi_rho", "eta_rho"),
        coords={"xi_rho": ds.xi_rho, "eta_rho": ds.eta_rho},
        attrs={"long_name": "Coriolis parameter at RHO-points", "units": "second-1", "field": "Corilis, scalar"},
    )
    ds["pm"] = xr.DataArray(
        data=np.nan,
        dims=("xi_rho", "eta_rho"),
        coords={"xi_rho": ds.xi_rho, "eta_rho": ds.eta_rho},
        attrs={"long_name": "curvilinear coordinate metric in XI", "units": "meter-1", "field": "pm, scalar"},
    )
    ds["pn"] = xr.DataArray(
        data=np.nan,
        dims=("xi_rho", "eta_rho"),
        coords={"xi_rho": ds.xi_rho, "eta_rho": ds.eta_rho},
        attrs={"long_name": "curvilinear coordinate metric in ETA", "units": "meter-1", "field": "pn, scalar"},
    )
    ds["dndx"] = xr.DataArray(
        data=np.nan,
        dims=("xi_rho", "eta_rho"),
        coords={"xi_rho": ds.xi_rho, "eta_rho": ds.eta_rho},
        attrs={"long_name": "xi derivative of inverse metric factor pn", "units": "meter", "field": "dndx, scalar"},
    )
    ds["dmde"] = xr.DataArray(
        data=np.nan,
        dims=("xi_rho", "eta_rho"),
        coords={"xi_rho": ds.xi_rho, "eta_rho": ds.eta_rho},
        attrs={"long_name": "eta derivative of inverse metric factor pm", "units": "meter", "field": "dmde, scalar"},
    )
    ds["x_rho"] = xr.DataArray(
        data=np.nan,
        dims=("xi_rho", "eta_rho"),
        coords={"xi_rho": ds.xi_rho, "eta_rho": ds.eta_rho},
        attrs={"long_name": "x location of RHO-points", "units": "meter"},
    )
    ds["y_rho"] = xr.DataArray(
        data=np.nan,
        dims=("xi_rho", "eta_rho"),
        coords={"xi_rho": ds.xi_rho, "eta_rho": ds.eta_rho},
        attrs={"long_name": "y location of RHO-points", "units": "meter"},
    )
    ds["x_psi"] = xr.DataArray(
        data=np.nan,
        dims=("xi_psi", "eta_psi"),
        coords={"xi_psi": ds.xi_psi, "eta_psi": ds.eta_psi},
        attrs={"long_name": "x location of PSI-points", "units": "meter"},
    )
    ds["y_psi"] = xr.DataArray(
        data=np.nan,
        dims=("xi_psi", "eta_psi"),
        coords={"xi_psi": ds.xi_psi, "eta_psi": ds.eta_psi},
        attrs={"long_name": "y location of PSI-points", "units": "meter"},
    )
    ds["x_u"] = xr.DataArray(
        data=np.nan,
        dims=("xi_u", "eta_u"),
        coords={"xi_u": ds.xi_u, "eta_u": ds.eta_u},
        attrs={"long_name": "x location of U-points", "units": "meter"},
    )
    ds["y_u"] = xr.DataArray(
        data=np.nan,
        dims=("xi_u", "eta_u"),
        coords={"xi_u": ds.xi_u, "eta_u": ds.eta_u},
        attrs={"long_name": "y location of U-points", "units": "meter"},
    )
    ds["x_v"] = xr.DataArray(
        data=np.nan,
        dims=("xi_v", "eta_v"),
        coords={"xi_v": ds.xi_v, "eta_v": ds.eta_v},
        attrs={"long_name": "x location of V-points", "units": "meter"},
    )
    ds["y_v"] = xr.DataArray(
        data=np.nan,
        dims=("xi_v", "eta_v"),
        coords={"xi_v": ds.xi_v, "eta_v": ds.eta_v},
        attrs={"long_name": "y location of V-points", "units": "meter"},
    )
    ds["lat_rho"] = xr.DataArray(
        data=np.nan,
        dims=("xi_rho", "eta_rho"),
        coords={"xi_rho": ds.xi_rho, "eta_rho": ds.eta_rho},
        attrs={"long_name": "latitude of RHO-points", "units": "degree_north"},
    )
    ds["lon_rho"] = xr.DataArray(
        data=np.nan,
        dims=("xi_rho", "eta_rho"),
        coords={"xi_rho": ds.xi_rho, "eta_rho": ds.eta_rho},
        attrs={"long_name": "longitude of RHO-points", "units": "degree_east"},
    )
    ds["lat_psi"] = xr.DataArray(
        data=np.nan,
        dims=("xi_psi", "eta_psi"),
        coords={"xi_psi": ds.xi_psi, "eta_psi": ds.eta_psi},
        attrs={"long_name": "latitude of PSI-points", "units": "degree_north"},
    )
    ds["lon_psi"] = xr.DataArray(
        data=np.nan,
        dims=("xi_psi", "eta_psi"),
        coords={"xi_psi": ds.xi_psi, "eta_psi": ds.eta_psi},
        attrs={"long_name": "longitude of PSI-points", "units": "degree_east"},
    )
    ds["lat_u"] = xr.DataArray(
        data=np.nan,
        dims=("xi_u", "eta_u"),
        coords={"xi_u": ds.xi_u, "eta_u": ds.eta_u},
        attrs={"long_name": "latitude of U-points", "units": "degree_north"},
    )
    ds["lon_u"] = xr.DataArray(
        data=np.nan,
        dims=("xi_u", "eta_u"),
        coords={"xi_u": ds.xi_u, "eta_u": ds.eta_u},
        attrs={"long_name": "longitude of U-points", "units": "degree_east"},
    )
    ds["lat_v"] = xr.DataArray(
        data=np.nan,
        dims=("xi_v", "eta_v"),
        coords={"xi_v": ds.xi_v, "eta_v": ds.eta_v},
        attrs={"long_name": "latitude of V-points", "units": "degree_north"},
    )
    ds["lon_v"] = xr.DataArray(
        data=np.nan,
        dims=("xi_v", "eta_v"),
        coords={"xi_v": ds.xi_v, "eta_v": ds.eta_v},
        attrs={"long_name": "longitude of V-points", "units": "degree_east"},
    )
    ds["mask_rho"] = xr.DataArray(
        data=np.nan,
        dims=("xi_rho", "eta_rho"),
        coords={"xi_rho": ds.xi_rho, "eta_rho": ds.eta_rho},
        attrs={"long_name": "mask on RHO-points", "option_0": "land", "option_1": "water"},
    )
    ds["mask_u"] = xr.DataArray(
        data=np.nan,
        dims=("xi_u", "eta_u"),
        coords={"xi_u": ds.xi_u, "eta_u": ds.eta_u},
        attrs={"long_name": "mask on U-points", "option_0": "land", "option_1": "water"},
    )
    ds["mask_v"] = xr.DataArray(
        data=np.nan,
        dims=("xi_v", "eta_v"),
        coords={"xi_v": ds.xi_v, "eta_v": ds.eta_v},
        attrs={"long_name": "mask on U-points", "option_0": "land", "option_1": "water"},
    )
    ds["mask_psi"] = xr.DataArray(
        data=np.nan,
        dims=("xi_psi", "eta_psi"),
        coords={"xi_psi": ds.xi_psi, "eta_psi": ds.eta_psi},
        attrs={"long_name": "mask on PSI-points", "option_0": "land", "option_1": "water"},
    )
    ds["angle"] = xr.DataArray(
        data=np.nan,
        dims=("xi_rho", "eta_rho"),
        coords={"xi_rho": ds.xi_rho, "eta_rho": ds.eta_rho},
        attrs={"long_name": "angle between xi axis and east", "units": "degree"},
    )

    try:
        ds.to_netcdf(the_roms_file, mode="w")
    except Exception as e:
        print(f" Unable to open ROMS NetCDF output file: {e}")
        return
    