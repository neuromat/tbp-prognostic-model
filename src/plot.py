import logging
import numpy as np
import utils
import json
import os
from pathlib import Path
from matplotlib import patches as mpatches, pyplot as plt
import pandas as pd
import seaborn

THRESHOLD_ACCURACY_FILE_PREFIX  = 'feature_threshold_vs_accuracy'
FEATURE_IMPORTANCE_FILE_PREFIX  = 'feature_importance'
FEATURE_CONTRIBUTION_FILE_PREFIX  = 'feature_contribution'

# Load plots' dictionaire
with open('plotsDictionary.json', 'r') as fp:
    plots_dictionaire = json.load(fp)

# Load attributes' dictionaire
with open('attributesDictionary.json', 'r') as fp:
    attr_dictionaire = json.load(fp)    
    

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        #if isinstance(obj, np.floating):
        #    return float(obj)
        #if isinstance(obj, np.ndarray):
        #    return obj.tolist()
        return super(NpEncoder, self).default(obj)


def create_model_plots(data_path, graphs_path):
    plot_function_caller(data_path, '**/**/**/'+THRESHOLD_ACCURACY_FILE_PREFIX+'_data__*.json', 
                            graphs_path,plot_feature_importance_vs_threshold_accuracy)
    plot_function_caller(data_path,'**/**/**/'+FEATURE_IMPORTANCE_FILE_PREFIX+'_data__*.json', 
                            graphs_path,plot_feature_importance)
    plot_function_caller(data_path,'**/**/**/'+FEATURE_CONTRIBUTION_FILE_PREFIX+'_data__*.json', 
                            graphs_path,plot_all_feature_contributions,True)

def plot_function_caller(data_path, file_name_pattern, graphs_path, plot_function, read_x=False):
    files_path = utils.findFiles(file_name_pattern, data_path)
    for file_path in files_path:
        with open(file_path, 'r') as fp:
            data = json.load(fp)

        if read_x:
            filename, file_extension = os.path.splitext(file_path)
            X = pd.read_csv(filename+'.csv')
            data['X'] = X

        model_path =  os.path.dirname(file_path).split(data_path)[1][1:]
        data['path'] = os.path.join(graphs_path, model_path)
        Path(data['path']).mkdir(parents=True, exist_ok=True)
        data['lang'] = 'en'
        plot_function(**data)
        data['lang'] = 'pt-br'
        plot_function(**data)


def save_feat_importance_vs_threshold_accuracy_data(path,xvalues,yvalues,class_name,special):
    data =  {'xvalues' : xvalues,'yvalues': yvalues,'class_name':class_name,'special':special}
    with open(os.path.join(path,THRESHOLD_ACCURACY_FILE_PREFIX+'_data__'+class_name+'.json'), 'w') as fp:
        json.dump(data, fp, cls = NpEncoder)


def plot_feature_importance_vs_threshold_accuracy(path,xvalues,yvalues,class_name,special,lang='en'):
    ax = plt.subplot(111)
    #ax.scatter(xvalues,yvalues,marker='x',s=60)
    ax.plot(xvalues,yvalues,'x-')
    if(special is not None):
        ax.scatter(xvalues[special],yvalues[special],marker='x',s=60,color='red',linewidths=3)

    # TODO não deixar fixo esse eixo y? o gráfico está ficando cortado
    #plt.axis((-0.005,0.007,0.6,0.85))
    plt.xlabel(plots_dictionaire['feature_importance_vs_threshold_accuracy']['xlabel'][lang])
    plt.ylabel(plots_dictionaire['feature_importance_vs_threshold_accuracy']['ylabel'][lang])
    plt.title(attr_dictionaire[class_name]['name'][lang] + '\n' + plots_dictionaire['feature_importance_vs_threshold_accuracy']['title'][lang])

    plt.tight_layout()
    plt.savefig(os.path.join(path,THRESHOLD_ACCURACY_FILE_PREFIX+'_plot_'+lang+'__'+class_name+'.png'))
    plt.close()


def save_feature_importance(path,collections,attributes,class_name,selected_features=False):
    data =  {'collections': collections ,'attributes' : attributes,'class_name':class_name,'selected_features':selected_features}
    class_name += '_selected_features' if selected_features else '_all_features'
    with open(os.path.join(path,FEATURE_IMPORTANCE_FILE_PREFIX+'_data__'+class_name+'.json'), 'w') as fp:
        json.dump(data, fp, cls = NpEncoder)

def plot_feature_importance(collections,attributes,path,class_name,selected_features=False, lang='en'):
    ax = plt.axes()

    translated_attributes = []
    for attr in attributes:
        translated_attributes.append(attr_dictionaire[attr]['name'][lang])

    ax.boxplot(collections)
    ax.set_xticklabels(translated_attributes,rotation=75,size='small')
    x1,x2,y1,y2 = plt.axis()
    if(y2 <= 0.01):
        y2 = 0.01
    plt.axis((x1,x2,y1,y2))
    plt.xlabel(plots_dictionaire['feature_importance']['xlabel'][lang])
    plt.ylabel(plots_dictionaire['feature_importance']['ylabel'][lang])
    
    title = attr_dictionaire[class_name]['name'][lang]
    if selected_features:
        title += plots_dictionaire['feature_importance']['selected_features'][lang]  
        class_name += '_selected_features'
    else: 
        title += plots_dictionaire['feature_importance']['all_features'][lang]
        class_name += '_all_features'

    plt.title(title + '\n' + plots_dictionaire['feature_importance']['title'][lang])

    plt.tight_layout()
    plt.savefig(os.path.join(path,FEATURE_IMPORTANCE_FILE_PREFIX+'_plot_'+lang+'__'+class_name+'.png'))
    plt.savefig(os.path.join(path,FEATURE_IMPORTANCE_FILE_PREFIX+'_plot_'+lang+'__'+class_name+'.svg'))
    plt.close()


def save_all_feature_contributions(path,X,y,fcs,class_of_interest,class_name):
    X.to_csv(os.path.join(path,FEATURE_CONTRIBUTION_FILE_PREFIX+'_data__'+class_name+'.csv'), index = False)
    data =  {'y' : list(y),'fcs' : fcs,'class_of_interest': class_of_interest,'path':path,'class_name':class_name}
    with open(os.path.join(path,FEATURE_CONTRIBUTION_FILE_PREFIX+'_data__'+class_name+'.json'), 'w') as fp:
        json.dump(data, fp, cls = NpEncoder)


def plot_all_feature_contributions(X,y,fcs,class_of_interest,path,class_name,lang='en'):
    for feature in X.columns:
        plot_feature_contributions1(X,y,feature,fcs,class_of_interest,path,class_name,lang)
        plot_feature_contributions2(X,y,feature,fcs,class_of_interest,path,class_name,lang)
        plot_feature_contributions3(X,y,feature,fcs,class_of_interest,path,class_name,lang)
        plot_feature_contributions4(X,y,feature,fcs,class_of_interest,path,class_name,lang)

def plot_feature_contributions1(X,y,feature,fcs,class_of_interest,path,class_name,lang='en'):
    plt.rc('axes', axisbelow=True)
    plt.grid(visible=True, which='both')
    x_graph = []
    colors = []
    y_graph = []

    found_contribution = False

    feature_index = str(X.columns.get_loc(feature))

    value = X[feature].values[utils.firstNotNan(X[:][feature])]
    isint = utils.isint(value)
    isfloat = utils.isfloat(value)
    if not isint and not isfloat:

        values = list(attr_dictionaire[feature]['values'].keys())

        if 'NINA' in values:
            # move nan to the end
            values.remove('NINA')
            values += [np.nan]   

        values.reverse()
        
        values_cont_blue_pos = [0] * len(values)
        values_cont_blue_neg = [0] * len(values)
        values_cont_red_pos = [0] * len(values)
        values_cont_red_neg = [0] * len(values)

        for i in range(X.shape[0]):
            if(feature_index in fcs[i].keys()):
                found_contribution = True

                x_graph.append(fcs[i][feature_index])
                y_graph.append(values.index(X[feature].values[i]))
                if(y[i] == class_of_interest):
                    colors.append('blue')
                    if x_graph[-1] > 0:
                        values_cont_blue_pos[y_graph[-1]] += 1
                    else:
                        values_cont_blue_neg[y_graph[-1]] += 1
                else:
                    colors.append('red')
                    if x_graph[-1] > 0:
                        values_cont_red_pos[y_graph[-1]] += 1
                    else:
                        values_cont_red_neg[y_graph[-1]] += 1

            # if(X[feature].values[i] not in contributions.keys()):
            #     contributions[X[feature].values[i]] = [fcs[i][feature_index]]
            # else:
            #     contributions[X[feature].values[i]].append(fcs[i][feature_index])
        
        if not found_contribution:
            # Leave the function without ploting the graph
            logging.info('feature \'%s\' has not entered in the model' %feature)
            return 
        

        if np.nan in values:
            i = values.index(np.nan)
            values[i] = 'NINA'

    else:
        if isint:
            values = sorted([round(int(i),4) for i in (set(X[:][feature])) if not utils.isnan(i)]) + [np.nan]
        else:
            values = sorted([round(float(i),4) for i in (set(X[:][feature])) if not utils.isnan(i)]) + [np.nan]

        values_cont_blue_pos = [0] * len(values)
        values_cont_blue_neg = [0] * len(values)
        values_cont_red_pos = [0] * len(values)
        values_cont_red_neg = [0] * len(values)

        for i in range(X.shape[0]):
            if(feature_index in fcs[i].keys()):
                found_contribution = True

                x_graph.append(fcs[i][feature_index])
                y_graph.append((X[feature].values[i]))
                if(y[i] == class_of_interest):
                    colors.append('blue')
                    if x_graph[-1] > 0:
                        values_cont_blue_pos[y_graph[-1]] += 1
                    else:
                        values_cont_blue_neg[y_graph[-1]] += 1
                else:
                    colors.append('red')
                    if x_graph[-1] > 0:
                        values_cont_red_pos[y_graph[-1]] += 1
                    else:
                        values_cont_red_neg[y_graph[-1]] += 1

        if not found_contribution:
            # Leave the function without ploting the graph
            logging.info('feature \'%s\' has not entered in the model' %feature)
            return 

    coi = str(class_of_interest)
    ax = plt.subplot(111)

    ax.set_title(attr_dictionaire[class_name]['name'][lang] + ' - ' +  attr_dictionaire[feature]['name'][lang] + 
                 '\n' + plots_dictionaire['feature_contributions']['title'][lang])
    ax.scatter(x_graph,y_graph,marker='x',s=60,facecolors=colors)
    plt.xlabel(plots_dictionaire['feature_contributions']['xlabel'][lang])
    plt.ylabel(plots_dictionaire['feature_contributions']['ylabel'][lang])

    translated_distinct_values = [attr_dictionaire[feature]['values'][value]['name'][lang] for value in values]

    ax.set_yticks(np.array(range(len(translated_distinct_values)+2))-1)
    labels = [str('')]+translated_distinct_values+[str('')]
    ax.set_yticklabels(labels)
    red_patch = mpatches.Patch(color='red')
    blue_patch = mpatches.Patch(color='blue')

    class_success = plots_dictionaire['feature_contributions']['legend']['success'][lang]
    class_unsuccess = plots_dictionaire['feature_contributions']['legend']['unsuccess'][lang]
    plt.legend([blue_patch, red_patch],[class_success, class_unsuccess],numpoints=1,fontsize='small',
                fancybox=True, title=plots_dictionaire['feature_contributions']['legend']['class'][lang])

    plt.tight_layout()
    plt.savefig(os.path.join(path,FEATURE_CONTRIBUTION_FILE_PREFIX+'_plot_'+lang+'__'+class_name+'__'+feature+'_1.png'))
    plt.savefig(os.path.join(path,FEATURE_CONTRIBUTION_FILE_PREFIX+'_plot_'+lang+'__'+class_name+'__'+feature+'_1.svg'))
    plt.close()


def plot_feature_contributions2(X,y,feature,fcs,class_of_interest,path,class_name,lang='en'):
    plt.rc('axes', axisbelow=True)
    plt.grid(visible=True, which='both')

    feature_index = str(X.columns.get_loc(feature))

    value = X[feature].values[utils.firstNotNan(X[:][feature])]
    isint = utils.isint(value)
    isfloat = utils.isfloat(value)
    if not isint and not isfloat:
        distinct_values = list(attr_dictionaire[feature]['values'].keys())

        if 'NINA' in distinct_values:
            # move nan to the end
            distinct_values.remove('NINA')
            distinct_values += [np.nan]   
        values_idx = [distinct_values.index(val) for val in X[:][feature]]

    else:
        if isint:
            distinct_values = sorted([round(int(val),4) for val in (set(X[:][feature])) if not utils.isnan(val)]) + [np.nan]
            values_idx = [distinct_values.index(round(int(val),4)) for val in X[:][feature]]
        else:
            distinct_values = sorted([round(float(val),4) for val in (set(X[:][feature])) if not utils.isnan(val)]) + [np.nan]
            values_idx = [distinct_values.index(round(float(val),4)) for val in X[:][feature]]

    
    n = len(distinct_values)
    cont_contrib_positive_success = [0] * n
    cont_contrib_positive_unsuccess = [0] * n
    cont_contrib_negative_success = [0] * n
    cont_contrib_negative_unsuccess = [0] * n

    found_feature = False
    for i in range(X.shape[0]):
        if(feature_index in fcs[i].keys()):
            found_feature = True
            if(y[i] == class_of_interest):
                if fcs[i][feature_index] > 0:
                    cont_contrib_positive_success[values_idx[i]] += 1
                else:
                    cont_contrib_negative_success[values_idx[i]] += 1
            else:
                if fcs[i][feature_index] > 0:
                    cont_contrib_positive_unsuccess[values_idx[i]] += 1
                else:
                    cont_contrib_negative_unsuccess[values_idx[i]] += 1

    if not found_feature:
        logging.info('feature \'%s\' has not entered in the model' %feature)
        return 

    ind = np.arange(n) # the x locations for the groups
    width = 0.25

    ax = plt.subplot(111)
 
    bar1 = ax.bar(ind-width/2, cont_contrib_positive_success, width, color='cornflowerblue') 
    bar2 = ax.bar(ind+width/2, cont_contrib_positive_unsuccess, width, color='tomato') 
    bar3 = ax.bar(ind-width/2, -np.array(cont_contrib_negative_success), width, color='cornflowerblue') 
    bar4 = ax.bar(ind+width/2, -np.array(cont_contrib_negative_unsuccess), width, color='tomato') 

    labels1 = [x if x != 0 else "" for x in cont_contrib_positive_success]
    labels2 = [x if x != 0 else "" for x in cont_contrib_positive_unsuccess]
    labels3 = [x if x != 0 else "" for x in cont_contrib_negative_success]
    labels4 = [x if x != 0 else "" for x in cont_contrib_negative_unsuccess]

    ax.bar_label(bar1, labels=labels1, label_type='center', fontweight='bold')
    ax.bar_label(bar2, labels=labels2, label_type='center', fontweight='bold')
    ax.bar_label(bar3, labels=labels3, label_type='center', fontweight='bold')
    ax.bar_label(bar4, labels=labels4, label_type='center', fontweight='bold')

    positive_max = max([max(cont_contrib_positive_success),max(cont_contrib_positive_unsuccess)])
    negative_max = -max([max(cont_contrib_negative_success),max(cont_contrib_negative_unsuccess)])
    yticks = [positive_max/2, 0 , negative_max/2]
    yticks_labels = [plots_dictionaire['feature_contributions']['legend']['positive_contrib'][lang],"",plots_dictionaire['feature_contributions']['legend']['negative_contrib'][lang]]

    ax.set_ylabel(plots_dictionaire['feature_contributions']['ylabel_histogram'][lang])
    ax.set_xlabel(plots_dictionaire['feature_contributions']['xlabel_histogram'][lang])
    ax.set_title(attr_dictionaire[class_name]['name'][lang] + ' - ' +  attr_dictionaire[feature]['name'][lang] + 
                 '\n' + plots_dictionaire['feature_contributions']['title'][lang])
    ax.set_xticks(ind)
    ax.set_yticks(yticks)
    
    if not isint and not isfloat and np.nan in distinct_values:
        i = distinct_values.index(np.nan)
        distinct_values[i] = 'NINA'

    translated_distinct_values = [attr_dictionaire[feature]['values'][value]['name'][lang] for value in distinct_values]
   
    ax.set_xticklabels(translated_distinct_values,rotation=45,size='small')
    ax.set_yticklabels(yticks_labels)

    ax.legend(labels=[plots_dictionaire['feature_contributions']['legend']['success'][lang],
            plots_dictionaire['feature_contributions']['legend']['unsuccess'][lang]], fontsize='small', fancybox=True)

    plt.axhline(linewidth=1, color='black')

    plt.tight_layout()
    plt.savefig(os.path.join(path,FEATURE_CONTRIBUTION_FILE_PREFIX+'_plot_'+lang+'__'+class_name+'__'+feature+'_2.png'))
    plt.savefig(os.path.join(path,FEATURE_CONTRIBUTION_FILE_PREFIX+'_plot_'+lang+'__'+class_name+'__'+feature+'_2.svg'))
    plt.close()


def plot_feature_contributions3(X,y,feature,fcs,class_of_interest,path,class_name,lang='en'):
    plt.rc('axes', axisbelow=True)
    plt.grid(visible=True, which='both')

    feature_index = str(X.columns.get_loc(feature))

    value = X[feature].values[utils.firstNotNan(X[:][feature])]
    isint = utils.isint(value)
    isfloat = utils.isfloat(value)
    if not isint and not isfloat:
        distinct_values = list(attr_dictionaire[feature]['values'].keys())

        if 'NINA' in distinct_values:
            # move nan to the end
            distinct_values.remove('NINA')
            distinct_values += [np.nan]   
        values_idx = [distinct_values.index(val) for val in X[:][feature]]

    else:
        if isint:
            distinct_values = sorted([round(int(val),4) for val in (set(X[:][feature])) if not utils.isnan(val)]) + [np.nan]
            values_idx = [distinct_values.index(round(int(val),4)) for val in X[:][feature]]
        else:
            distinct_values = sorted([round(float(val),4) for val in (set(X[:][feature])) if not utils.isnan(val)]) + [np.nan]
            values_idx = [distinct_values.index(round(float(val),4)) for val in X[:][feature]]

    
    n = len(distinct_values)
    cont_contrib_positive_success = [0] * n
    cont_contrib_positive_unsuccess = [0] * n
    cont_contrib_negative_success = [0] * n
    cont_contrib_negative_unsuccess = [0] * n

    found_feature = False
    for i in range(X.shape[0]):
        if(feature_index in fcs[i].keys()):
            found_feature = True
            if(y[i] == class_of_interest):
                if fcs[i][feature_index] > 0:
                    cont_contrib_positive_success[values_idx[i]] += 1
                else:
                    cont_contrib_negative_success[values_idx[i]] += 1
            else:
                if fcs[i][feature_index] > 0:
                    cont_contrib_positive_unsuccess[values_idx[i]] += 1
                else:
                    cont_contrib_negative_unsuccess[values_idx[i]] += 1

    if not found_feature:
        logging.info('feature \'%s\' has not entered in the model' %feature)
        return 

    ind = np.arange(n) # the x locations for the groups
    width = 0.25

    ax = plt.subplot(111)
    bar1 = ax.bar(ind-width/2, cont_contrib_negative_success, width, color='cornflowerblue', hatch='//') 
    bar2 = ax.bar(ind-width/2, cont_contrib_negative_unsuccess, width,bottom=cont_contrib_negative_success, color='tomato', hatch='//') 
    bar3 = ax.bar(ind+width/2, cont_contrib_positive_success, width, color='cornflowerblue') 
    bar4 = ax.bar(ind+width/2, cont_contrib_positive_unsuccess, width,bottom=cont_contrib_positive_success, color='tomato') 

    labels1 = [x if x != 0 else "" for x in cont_contrib_negative_success]
    labels2 = [x if x != 0 else "" for x in cont_contrib_negative_unsuccess]
    labels3 = [x if x != 0 else "" for x in cont_contrib_positive_success]
    labels4 = [x if x != 0 else "" for x in cont_contrib_positive_unsuccess]
    ax.bar_label(bar1, labels=labels1, label_type='center', fontweight='bold')
    ax.bar_label(bar2, labels=labels2, label_type='center', fontweight='bold')
    ax.bar_label(bar3, labels=labels3, label_type='center', fontweight='bold')
    ax.bar_label(bar4, labels=labels4, label_type='center', fontweight='bold')

    ax.set_ylabel(plots_dictionaire['feature_contributions']['ylabel_histogram'][lang])
    ax.set_xlabel(plots_dictionaire['feature_contributions']['xlabel_histogram'][lang])
    ax.set_title(attr_dictionaire[class_name]['name'][lang] + ' - ' +  attr_dictionaire[feature]['name'][lang] + 
                 '\n' + plots_dictionaire['feature_contributions']['title'][lang])
    ax.set_xticks(ind)

    if not isint and not isfloat and np.nan in distinct_values:
        i = distinct_values.index(np.nan)
        distinct_values[i] = 'NINA'

    translated_distinct_values = [attr_dictionaire[feature]['values'][value]['name'][lang] for value in distinct_values]
   
    ax.set_xticklabels(translated_distinct_values,rotation=45,size='small')

    ax.legend(labels=[plots_dictionaire['feature_contributions']['legend']['negative_contrib_success'][lang],
            plots_dictionaire['feature_contributions']['legend']['negative_contrib_unsuccess'][lang],
            plots_dictionaire['feature_contributions']['legend']['positive_contrib_success'][lang],
            plots_dictionaire['feature_contributions']['legend']['positive_contrib_unsuccess'][lang]], fontsize='small', fancybox=True)
            
    plt.tight_layout()
    plt.savefig(os.path.join(path,FEATURE_CONTRIBUTION_FILE_PREFIX+'_plot_'+lang+'__'+class_name+'__'+feature+'_3.png'))
    plt.savefig(os.path.join(path,FEATURE_CONTRIBUTION_FILE_PREFIX+'_plot_'+lang+'__'+class_name+'__'+feature+'_3.svg'))
    plt.close()





def plot_feature_contributions4(X,y,feature,fcs,class_of_interest,path,class_name,lang='en'):
    
    df = pd.DataFrame()

    feature_column =  attr_dictionaire[feature]['name'][lang]
    contribution_column = plots_dictionaire['feature_contributions']['ylabel'][lang]
    class_column = plots_dictionaire['feature_contributions']['legend']['class'][lang]

    df[feature_column] = X[feature]
    df[class_column] = y
    df[contribution_column] = np.nan

    found_contribution = False
    feature_index = str(X.columns.get_loc(feature))
   
    for i in range(X.shape[0]):
        if(feature_index in fcs[i].keys()):
            found_contribution = True
            df.at[i,contribution_column] = fcs[i][feature_index]
    
    if not found_contribution:
        # Leave the function without ploting the graph
        logging.info('feature \'%s\' has not entered in the model' %feature)
        return 
    
    df[feature_column].fillna('NINA', inplace=True) 
    df[feature_column] = df[feature_column].apply(lambda x: attr_dictionaire[feature]['values'][x]['name'][lang])
    df[class_column] = df[class_column].apply(lambda x: attr_dictionaire[class_name]['values'][str(int(x))]['name'][lang])
 
    seaborn.set(style='whitegrid')
    sp = seaborn.swarmplot(x=contribution_column, y=feature_column, hue=class_column, data=df, palette="deep")
    sp.set_ylabel(plots_dictionaire['feature_contributions']['ylabel'][lang])
    sp.set_xlabel(plots_dictionaire['feature_contributions']['xlabel'][lang])
    sp.set_title(attr_dictionaire[class_name]['name'][lang] + ' - ' +  attr_dictionaire[feature]['name'][lang] + 
                 '\n' + plots_dictionaire['feature_contributions']['title'][lang])

    plt.tight_layout()
    fig = sp.get_figure()
    fig.savefig(os.path.join(path,FEATURE_CONTRIBUTION_FILE_PREFIX+'_plot_'+lang+'__'+class_name+'__'+feature+'_4.png')) 
    fig.savefig(os.path.join(path,FEATURE_CONTRIBUTION_FILE_PREFIX+'_plot_'+lang+'__'+class_name+'__'+feature+'_4.svg')) 
    plt.close()

    sp = seaborn.stripplot(x=contribution_column, y=feature_column, hue=class_column, data=df, palette="deep")
    sp.set_ylabel(plots_dictionaire['feature_contributions']['ylabel'][lang])
    sp.set_xlabel(plots_dictionaire['feature_contributions']['xlabel'][lang])
    sp.set_title(attr_dictionaire[class_name]['name'][lang] + ' - ' +  attr_dictionaire[feature]['name'][lang] + 
                 '\n' + plots_dictionaire['feature_contributions']['title'][lang])

    plt.tight_layout()
    fig = sp.get_figure()
    fig.savefig(os.path.join(path,FEATURE_CONTRIBUTION_FILE_PREFIX+'_plot_'+lang+'__'+class_name+'__'+feature+'_5.png')) 
    fig.savefig(os.path.join(path,FEATURE_CONTRIBUTION_FILE_PREFIX+'_plot_'+lang+'__'+class_name+'__'+feature+'_5.svg')) 
    plt.close()


if __name__ == '__main__':
    data_path = ('/home/kellyrb/Insync/kellyrb@ime.usp.br/Google Drive/Insync/florestas_classificador_plugin3/output_2023_08_30/data')
    graphs_path = ('/home/kellyrb/Insync/kellyrb@ime.usp.br/Google Drive/Insync/florestas_classificador_plugin3/output/graphs')
    #data_path = ('/home/kellyrb/Insync/kellyrb@ime.usp.br/Google Drive/Insync/florestas_grafos/data')
    #graphs_path = ('/home/kellyrb/Insync/kellyrb@ime.usp.br/Google Drive/Insync/florestas_grafos/graphs')

    create_model_plots(data_path,graphs_path)