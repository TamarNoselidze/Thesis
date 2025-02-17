import json

with open('Random_Position_Patch/classes.txt', 'r') as file:  
    lines = file.readlines()

class_mapping = {}

for line in lines:
    if not line.strip():
        continue
    
    parts = line.split('|')
    key = parts[1].strip()  
    value = parts[2].strip() 
    class_mapping[int(key)] = value  

with open('Random_Position_Patch/class_mapping.json', 'w') as json_file:  
    json.dump(class_mapping, json_file, indent=4)  