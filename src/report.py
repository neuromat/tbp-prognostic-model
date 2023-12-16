import json
import glob
import json
import os
import pandas as pd

def generateModelReports(languages=["pt-br","en"]):
    json_file_paths = glob.glob("output/models/*/*/*metrics.json")
    # Example of model metrics' file name:
    # final_prognostic_model_Pain_outcome_metrics.json
    prefix = "final_prognostic_model_"
    sufix = "_metrics.json"

    for lang in languages:
        perf_data = {"model": [], "group/origin" : []}
        initial_attr_data = {"model": [], "group/origin" : []}
        selected_attr_data = {"model": [], "group/origin" : []}
        missing_attr_data = {"model": [], "group/origin" : []}
        attr_importance_data = {"model": [], "group/origin" : []}
        
        # Load attributes' dictionaire
        with open('attributesDictionary.json', 'r') as fp:
            attr_dictionaire = json.load(fp)

        for a in attr_dictionaire.keys():
            tr_a =  attr_dictionaire[a]['name'][lang]
            selected_attr_data[tr_a] = []
            initial_attr_data[tr_a] = []
            missing_attr_data[tr_a] = []
            attr_importance_data[tr_a] = []

        for file_path in json_file_paths:
            outcome = os.path.basename(file_path)[len(prefix):-len(sufix)]
            group_origin = os.path.dirname(file_path)[len("output/models/"):]
            file = open(file_path, 'r')
            metrics = json.load(file)
            perf_data["model"].append(outcome)
            perf_data["group/origin"].append(group_origin)
            initial_attr_data["model"].append(outcome)
            initial_attr_data["group/origin"].append(group_origin)
            selected_attr_data["model"].append(outcome)
            selected_attr_data["group/origin"].append(group_origin)
            missing_attr_data["model"].append(outcome)
            missing_attr_data["group/origin"].append(group_origin)
            attr_importance_data["model"].append(outcome)
            attr_importance_data["group/origin"].append(group_origin)

            for k,v in metrics.items():
                if k not in perf_data:
                    perf_data[k] = []
                perf_data[k].append(str(v))

            initial_attributes = metrics['initial attributes, % of missing']
            importance = metrics['initial attributes, importance']
            for a in attr_dictionaire.keys():
                tr_a =  attr_dictionaire[a]['name'][lang]

                if a in initial_attributes.keys():
                    initial_attr_data[tr_a].append("X")
                    missing_attr_data[tr_a].append(initial_attributes[a])
                    attr_importance_data[tr_a].append(importance[a])
                else:
                    initial_attr_data[tr_a].append("")
                    missing_attr_data[tr_a].append("")
                    attr_importance_data[tr_a].append("")

                if a in metrics["selected attributes"]:
                    selected_attr_data[tr_a].append("X")
                else:
                    selected_attr_data[tr_a].append("")
            
            

        df1 = pd.DataFrame(perf_data).sort_values(by=['model','group/origin'])
        for a in attr_dictionaire.keys():
            tr_a =  attr_dictionaire[a]['name'][lang]
            df1 = df1.replace(a,tr_a,regex=True)

        df2 = pd.DataFrame(initial_attr_data).sort_values(by=['model','group/origin'])
        df3 = pd.DataFrame(selected_attr_data).sort_values(by=['model','group/origin'])
        df4 = pd.DataFrame(missing_attr_data).sort_values(by=['model','group/origin'])
        df5 = pd.DataFrame(attr_importance_data).sort_values(by=['model','group/origin'])

        with pd.ExcelWriter('output/performance_report_' + lang + '.xlsx') as writer:  
            df1.to_excel(writer, sheet_name="Performance Metrics", index = False)
            df2.to_excel(writer, sheet_name="Initial Attributes", index = False)
            df3.to_excel(writer, sheet_name="Selected Attributes", index = False)
            df4.to_excel(writer, sheet_name="Missing Rates (%)", index = False)
            df5.to_excel(writer, sheet_name="Attributes' Importance", index = False)


if __name__ == '__main__':
   generateModelReports() 