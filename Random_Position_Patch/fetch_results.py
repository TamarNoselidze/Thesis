import argparse
import wandb
import csv

target_classes = {
    153: "Maltese Dog",
    309: "Bee",
    481: "Cassette",
    746: "Hockey Puck",
    932: "Pretzel"
}

models = {
    "vit_b_16": "VB/16",
    "vit_b_32": "VB/32",
    "vit_l_16": "VL/16",
    "swin_b": "Sw-B",
    "resnet50": "RN50",
    "resnet152": "RN152",
    "vgg16_bn": "VGG"
}


ENTITY = "takonoselidze-charles-university"

api = wandb.Api()


def fetch_rows(attack_type):

    rows = []

    for class_id, class_name in target_classes.items():
        #  5 rows per class for iteration 1–5
        iteration_data = {i: [class_name, i] for i in range(1, 6)}

        for model_key, _ in models.items():
            project_name = f"F train {attack_type} ={class_id}=  {model_key}"
            runs = api.runs(f"{ENTITY}/{project_name}")
            for i in range(1, 6):
                if i <= len(runs):
                    run = runs[i - 1]
                    try:
                        best_asr = f'{run.summary.get("summary/best_ASR"):.2f}'
                    except:
                        best_asr = "-"
                    iteration_data[i].append(best_asr)
                else:
                    iteration_data[i].append("-")            

        rows.extend(iteration_data.values())
    
    return rows

def average_rows(rows):
    from collections import defaultdict

    # Structure: {class_name: {model_index: [list of ASR values as float]}}
    class_model_asrs = defaultdict(lambda: defaultdict(list))

    for row in rows:
        class_name = row[0]
        # row[2:] contains ASR values for models
        for i, val in enumerate(row[2:]):
            if val != "-":
                try:
                    class_model_asrs[class_name][i].append(float(val))
                except ValueError:
                    pass  # skip invalid floats

    # Create averaged rows
    averaged_rows = []
    for class_name in target_classes.values():
        averaged_row = [class_name]
        for i in range(len(models)):
            vals = class_model_asrs[class_name][i]
            if vals:
                avg = sum(vals) / len(vals)
                averaged_row.append(f"{avg:.2f}")
            else:
                averaged_row.append("-")
        averaged_rows.append(averaged_row)

    return averaged_rows


def write_to_output(rows, averaged, output_file):
    with open(output_file, mode="w", newline="") as f:
        writer = csv.writer(f)

        if not averaged:
            header = ["Target class", "Itr #"] + list(models.values())
        else: 
            header = ["Target Class"] + list(models.values())
        writer.writerow(header)
        writer.writerows(rows)



if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Random Position Patch Attack')
    parser.add_argument('--attack_type')
    parser.add_argument('--averaged', default=False)
    parser.add_argument('--output_file', help='Path to the output file where generators will be saved.', default='./wandb_asr_table.csv')

    args = parser.parse_args()

    rows = fetch_rows(args.attack_type)
    if args.averaged:
        rows = average_rows(rows)

    write_to_output(rows, args.averaged, args.output_file)