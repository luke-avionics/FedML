import wandb
import csv

wandb.init(
    # project="federated_nas",
    project="cvpr-plotter",
    name="fig1"
)


x=[]
y=[]
data_path="/home/yz87/Downloads/wandb_export_2020-11-15T01_32_40.994-06_00.csv"

with open(data_path, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for idx, row in enumerate(reader):
        if idx ==0:
            continue
        elif idx ==1:
            for _ in row[0].split('","'):
                x.append([])
        for item_id,item in enumerate(row[0].split(',')):
            #print(item_id)
            if item.replace('"','').replace('"','') is not '': 
                x[item_id].append(float(item.replace('"','').replace('"','')))
#print(x)



# x=[]
# y=[]


for idx,x_n in enumerate(x[0]):
    print(x_n)
    if idx<len(x[1]):
        wandb.log({"Test/Acc": x[1][idx],  "round": x[0][idx]},commit=False)
    if idx<len(x[2]):
        wandb.log({"Test/Acc1": x[2][idx], "round": x[0][idx]})
