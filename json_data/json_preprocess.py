import json
import os
from tqdm import tqdm

def preprocess(infile, outfile, json_list , type='aphid'):
    for file in tqdm(json_list):
        with open(f'{infile}/{file}') as json_file:
            json_data = json.load(json_file)

        ### label change
        for shape in json_data['shapes']:
            shape['label'] = type

        ### bounding box generate
        for shape in json_data['shapes']:
            points = shape['points']
            x_coordinates = [point[0] for point in points]
            y_coordinates = [point[1] for point in points]
            min_x = min(x_coordinates)
            max_x = max(x_coordinates)
            min_y = min(y_coordinates)
            max_y = max(y_coordinates)

            shape['points'] = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]


        ### update json file
        with open(f'{outfile}/{file}', 'w') as json_file:
            json.dump(json_data, json_file)


def main():
    infile = r'Z:\Projects\2307_2308_bug\model_input\annotated\train\merged_train'
    outfile = './output/aphid'

    if not os.path.exists(outfile):
        os.makedirs(outfile)

    label_list = [x for x in os.listdir(infile) if x.endswith(".json")]
    preprocess(infile, outfile, label_list)

if __name__=='__main__':
    main()