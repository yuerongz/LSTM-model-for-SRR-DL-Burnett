import csv


def read_csv(csvfile_name):
    output = []
    with open(csvfile_name, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            output.append(row)
    return output


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


