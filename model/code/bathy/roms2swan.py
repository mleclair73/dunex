import xarray as xr
import numpy as np

# Note(@mleclair) This does not work, use the matlab version for now and debug.

def roms2swan(input_data):
    """
    Function to convert a ROMS netCDF depth grid into a SWAN ASCII depth grid
    and create the SWAN curvilinear coordinates grid.

    Parameters:
    input_data (str or xarray.Dataset): Either the path to a netCDF file or an xarray.Dataset
                                        containing the necessary variables: lon_rho, lat_rho, h, mask_rho.

    Returns:
    None (but creates two ASCII files: swan_bathy.bot and swan_coord.grd)
    """
    if isinstance(input_data, str):
        # Load data from netCDF file
        ds = xr.open_dataset(input_data)
        x_rho = ds['lon_rho'].values
        y_rho = ds['lat_rho'].values
        h = ds['h'].values
        mask_rho = ds['mask_rho'].values
    elif isinstance(input_data, xr.Dataset):
        x_rho = input_data['lon_rho'].values
        y_rho = input_data['lat_rho'].values
        h = input_data['h'].values
        mask_rho = input_data['mask_rho'].values
    else:
        raise ValueError("Input must be either a string (file path) or an xarray.Dataset")

    # Replace land positions with the flag for land (9999)
    land_values = np.where(mask_rho == 0)
    h[land_values] = 9999

    # Print depths to the bathymetry file
    with open('swan_bathy.bot', 'w') as f:
        for row in h:
            f.write('   ' + ' '.join(f'{val:12.8f}' for val in row) + '\n')

    # Print grid coordinates to the grid file
    with open('swan_coord.grd', 'w') as f:
        for row in x_rho:
            f.write(' '.join(f'{val:18.12f}' for val in row) + '\n')
        for row in y_rho:
            f.write(' '.join(f'{val:18.12f}' for val in row) + '\n')

    print("I created swan_coord.grd and swan_bathy.bot: these are part of INPUT.")