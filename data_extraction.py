import csv
import numpy as np
from netCDF4 import Dataset
import geopandas
from time import time


def coords2rc(transform, coords):
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = -transform[5]
    col = int((coords[0] - xOrigin) / pixelWidth)
    row = int((yOrigin - coords[1]) / pixelHeight)
    return row, col


def read_shp_point(filename):
    """Read the Point shapefile attributes as a list of xy coordinates"""
    result = list()
    startPoints = geopandas.read_file(filename)
    for pt in startPoints.geometry:
        result.append((pt.x, pt.y))
    return result


def getNCtransform(dataset):
    xres = dataset.variables['x'][1].data - dataset.variables['x'][0].data
    xmin = dataset.variables['x'][0].data - xres / 2  # calculate the original extent xmin
    yres = dataset.variables['y'][-1].data - dataset.variables['y'][-2].data
    ymax = dataset.variables['y'][-1].data + yres / 2  # calculate the original extent ymax
    geotransform = (xmin, xres, 0, ymax, 0, -yres)
    return geotransform


def read_csv(csvfile_name):
    output = []
    with open(csvfile_name, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            output.append(row)
    return output


def save_csv(out_file, out_ls):
    # out_ls is a nested list oriented with [row][col]
    with open(out_file, mode='w', newline='') as target_file:
        csv_writer = csv.writer(target_file, delimiter=',')
        csv_writer.writerows(out_ls)
    return 0


def build_event_name_dict(csvfile_name):
    csv_ls = read_csv(csvfile_name)
    output = dict()
    headers = csv_ls[0]
    for row in csv_ls[1:]:
        item_info = dict()
        for item_no in range(len(row)):
            item_info[headers[item_no]] = row[item_no]
        output[row[1]] = item_info  # use event name as main key
    return output


def extract_15min_q(event_input_file, event_peak_hour):
    input_ls = read_csv(event_input_file)
    # extract_columns = ['TimeU', 'Paradise_Outflow', 'TimeSL', 'SeaLev']
    # extract_col = [0, 1]
    time_steps = np.arange(0.25, event_peak_hour, 0.25)     # cut-off to peak Q, extend further to allow water travel
    input_t = [float(i) for i in list(list(zip(*input_ls[1:]))[0])]
    input_q = [float(i) for i in list(list(zip(*input_ls[1:]))[1])]
    q_15min = [input_q[input_t.index(curr_step)] for curr_step in time_steps]
    return q_15min


def save_rawdata_to_csv(raw_data, work_dir, grid_no):
    # for each point, save a csv
    for pt_coords in raw_data.keys():
        headers_ls = list(raw_data[pt_coords].keys())
        cols_ls = list(raw_data[pt_coords].values())
        out_csv = [[headers_ls[i]] + cols_ls[i] for i in range(len(headers_ls))]
        save_csv(f"{work_dir}grid_{grid_no}_rawdata.csv", out_csv)
    return 0


def extract_raw_data(work_dir, event_dir, event_input_dir, event_info,
                     pts_file, grid_no):
    """
    For one event as provided, extract
    1. the output data sequences for each points provided in pts_file
    2. the input data sequences
    In total, event_no * point_no tables will be built.
    :param pts_file: shp file name
    :return: raw data table, headers=[]
    """
    starttime = time()
    # event_peak_hours = {'1971':198.25, '2010':158.75, '2013':82.25, 'design':96.63}  # event_info['eventname]['Hydro_pattern']
    event_peak_hours = {'1971':430, '2010':280, '2013':190, 'design':240}  # event_info['eventname]['Hydro_pattern']
    pts_coords = read_shp_point(pts_file)
    out_csv = dict()
    curr_pt = pts_coords[grid_no]
    out_csv[curr_pt] = dict()
    evecount = 1
    for curr_event in event_info.keys():  # for each event
        print(evecount)
        evecount += 1
        ncfile = f"{event_dir}{curr_event}/Paradise_{curr_event}_002.nc"
        nc_data = Dataset(ncfile)
        # take out data with 15 min interval
        event_Q_15min = extract_15min_q(f"{event_input_dir}{event_info[curr_event]['Inputname']}.csv",
                                        event_peak_hours[event_info[curr_event]['Hydro_pattern']])
        pts_rc = coords2rc(getNCtransform(nc_data), curr_pt)  # (r, c)
        flipped_r = nc_data.variables['water_level'].shape[1] - pts_rc[0] - 1   # NC array to GDAL array, row flip
        wl_arr = nc_data.variables['water_level'][1:len(event_Q_15min) + 1, flipped_r, pts_rc[1]]
        wl_data = wl_arr.data
        wl_data[wl_arr.mask] = -999     # mark no-data as -999
        event_wl_15min = list(wl_data)
        out_csv[curr_pt][f"{curr_event}_Q"] = event_Q_15min
        out_csv[curr_pt][f"{curr_event}_WL"] = event_wl_15min
        print(time()-starttime)
    save_rawdata_to_csv(out_csv, work_dir, grid_no)
    write_pt_info_csv(work_dir, out_csv, grid_no)
    return out_csv


def extract_15min_cf(event_input_file, event_peak_hour, event_hydro_pattern):
    input_ls = read_csv(event_input_file)
    cf_15min = []
    input_t = [float(i) for i in list(list(zip(*input_ls[1:]))[2]) if bool(i)]
    time_steps = np.arange(0.25, event_peak_hour, 0.25)     # cut-off to peak Q, extend further to allow water travel
    if event_hydro_pattern == 'design':     # 3h linear interpolation
        for cf_no in range(3, 16):
            input_cf = [float(i) for i in list(list(zip(*input_ls[1:]))[cf_no]) if bool(i)]    # whole column
            linear_interp = list(np.interp(time_steps, input_t, input_cf))
            cf_15min.append(linear_interp)
    else:   # 1971 2010 2013 = 15min
        for cf_no in range(3, 16):
            input_cf = [float(i) for i in list(list(zip(*input_ls[1:]))[cf_no]) if bool(i)]
            cf_15min.append([input_cf[input_t.index(curr_step)] for curr_step in time_steps])
    return cf_15min


def extract_15min_sl(event_input_file, event_peak_hour):
    input_ls = read_csv(event_input_file)
    time_steps = np.arange(0.25, event_peak_hour, 0.25)     # cut-off to peak Q, extend further to allow water travel
    input_t = [float(i) for i in list(list(zip(*input_ls[1:]))[17]) if bool(i)]
    input_sl = [float(i) for i in list(list(zip(*input_ls[1:]))[16]) if bool(i)]
    sl_15min = list(np.interp(time_steps, input_t, input_sl))
    return sl_15min


def add_cf_sl_to_rawdata(rawdata, event_input_dir, event_info, grid_no):
    event_peak_hours = {'1971': 430, '2010': 280, '2013': 190, 'design': 240}  # event_info['eventname]['Hydro_pattern']
    # pts_coords = read_shp_point(pts_file)
    evecount = 1
    for curr_event in event_info.keys():  # for each event
        print(evecount)
        evecount += 1
        # take out data with 15 min interval
        event_CF_15min = extract_15min_cf(f"{event_input_dir}{event_info[curr_event]['Inputname']}.csv",
                                          event_peak_hours[event_info[curr_event]['Hydro_pattern']],
                                          event_info[curr_event]['Hydro_pattern'])
        event_SL_15min = extract_15min_sl(f"{event_input_dir}{event_info[curr_event]['Inputname']}.csv",
                                          event_peak_hours[event_info[curr_event]['Hydro_pattern']])
        for curr_pt in rawdata.keys():  # for each selected grid
            # print(grid_no)
            for cf_no in range(13):
                rawdata[curr_pt][f"{curr_event}_{cf_no}_CF"] = event_CF_15min[cf_no]
            rawdata[curr_pt][f"{curr_event}_SL"] = event_SL_15min
    save_rawdata_to_csv(rawdata, work_dir, grid_no)
    return rawdata


def write_pt_info_csv(work_dir, raw_data, grid_no):
    with open(f"{work_dir}grid_coords_{grid_no}.csv", mode='w', newline='') as ptfile:
        writer = csv.writer(ptfile, delimiter=',')
        grid_nos = 0
        out_ls = [['grid_No', 'X', 'Y']]
        for x_coord, y_coord in raw_data.keys():
            curr_ls = [grid_nos, x_coord, y_coord]
            out_ls.append(curr_ls)
            grid_nos += 1
        writer.writerows(out_ls)
    return 0


def read_rawdata_from_csv(work_dir, grid_no):
    raw_data = dict()
    with open(f"{work_dir}grid_coords_{grid_no}.csv", newline='') as ptfile:
        reader = csv.reader(ptfile)
        grids_info = []
        row_no = 0
        for row in reader:
            if row_no == 0:
                row_no += 1
                continue
            else:
                row_no += 1
                grids_info.append(tuple([float(i) for i in row[1:]]))
    with open(f"{work_dir}grid_{grid_no}_rawdata.csv", newline='') as rawdatafile:
        reader = csv.reader(rawdatafile)
        raw_data[grids_info[0]] = dict()
        for row in reader:
            raw_data[grids_info[0]][row[0]] = [float(i) for i in row[1:]]
    return raw_data


def fetch_raw_data(data_dir, grid_no):
    return read_rawdata_from_csv(data_dir, grid_no)


def fetch_event_info(event_info_file):
    event_summary_file = event_info_file
    event_info = build_event_name_dict(event_summary_file)
    return event_info


