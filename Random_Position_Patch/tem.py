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



rows = []

for class_id, class_name in target_classes.items():
    iteration_data = {i: [class_name if i == 1 else "", i] for i in range(1, 6)}
    
    for model_key, model_label in models.items():
        project_name = f"F train gpatch ={class_id}=  {model_key}"
        try:
            runs = api.runs(f"{ENTITY}/{project_name}")
            for i, run in enumerate(runs[:5]):  #  max 5 runs per model
                try:
                    best_asr = f'{run.summary.get("summary/best_ASR"):.2f}%'
                # if isinstance(summary_list, list) and i < len(summary_list):
                #     best_asr = summary_list[i].get("best ASR", "")
                except:
                    best_asr = "-"
                iteration_data[i + 1].append(best_asr)
        except Exception as e:
            print(f"Error with {project_name}: {e}")
            for i in range(1, 6):
                iteration_data[i].append("")
    
    rows.extend(iteration_data.values())



with open("wandb_asr_table.csv", mode="w", newline="") as f:
    writer = csv.writer(f)
    header = ["Target class", "Itr #"] + list(models.values())
    writer.writerow(header)
    writer.writerows(rows)
