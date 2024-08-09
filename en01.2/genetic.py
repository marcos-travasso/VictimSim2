import csv
import random

victims = {}
BASE = (65, 22)

with open('cluster4.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in reader:
        if row[0] == "ID":
            continue
        victim = {"id": int(row[0])}
        victim["coords"] = eval(f"{row[1:3]}".replace('"', "").replace("'",""))[0]
        victim["coords"] = (victim["coords"][0] + BASE[0], victim["coords"][1] + BASE[1])
        victim["feat"] = eval(f"{row[3:9]}".replace('"', "").replace("'",""))[0]
        victims[victim["id"]] = victim


shuffled_list = list(victims.keys())
random.shuffle(shuffled_list)
victims_list = [victims[x] for x in shuffled_list]
print(victims_list)