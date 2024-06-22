import numpy as np
def convert_person_name(item):
    # get a mapping
    selected_names = np.random.choice(PERSON_NAMES, 3, replace=False)

    item["context"] = item["context"].replace("PersonX", selected_names[0]).replace("PersonY", selected_names[1]).replace("PersonZ", selected_names[2])
    for i in range(len(item['candidates'])):
        item['candidates'][i] = item['candidates'][i].replace("PersonX", selected_names[0]).replace("PersonY", selected_names[1]).replace("PersonZ", selected_names[2])
    return item