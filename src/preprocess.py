import pandas as pd
import numpy as np
import csv
from pathlib import Path
import os
import utils
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport

DEBUG = 1

OPERATION_TRAINING = 0
OPERATION_BULK_CLASSIFICATION = 1
OPERATION_SINGLE_CLASSIFICATION = 2

def preprocess(input_path, output_path, operation=OPERATION_TRAINING, participant_code=None, data_reports=True):

    # Read responses for UAA, UFA, and USE questionnaires in dataframes
    if operation == OPERATION_SINGLE_CLASSIFICATION and participant_code is not None:
        # get data_path of per participant data
        uaa_path = utils.findFiles( '**/Per_participant*/Participant_'+participant_code+'/**/Q*_unified_admission_assessment_en.csv', input_path)
        if len(uaa_path) == 0:
            uaa_path = utils.findFiles( '**/Per_participant*/Participant_'+participant_code+'/**/Q*_avaliacao-de-entrada-unificada_pt-BR.csv', input_path)
            if len(uaa_path) == 0:
                print("Error: the Unified Admission Assessment response for patient %s was not found" %participant_code)
                return 
            
    else: # operation == OPERATION_TRAINING or operation == OPERATION_BULK_CLASSIFICATION
        # get data_path of per questionnaire data
        uaa_path = utils.findFiles('**/Per_questionnaire*/**/Q*_unified_admission_assessment_en.csv', input_path)
        if len(uaa_path) == 0 :  
            uaa_path = utils.findFiles('**/Per_questionnaire*/**/Q*_avaliacao-de-entrada-unificada_pt-BR.csv', input_path)

    ufa_path = utils.findFiles('**/Per_questionnaire*/**/Q*_unified_followup_assessment_en.csv', input_path)
    if len(ufa_path) == 0 :
        ufa_path = utils.findFiles('**/Per_questionnaire*/**/Q*_avaliacao-de-seguimento-unificada_pt-BR.csv', input_path)
    use_path = utils.findFiles('**/Per_questionnaire*/**/Q*_surgical_evaluation_en.csv', input_path)
    if len(use_path) == 0 :
        use_path = utils.findFiles('**/Per_questionnaire*/**/Q*_avaliacao-cirurgica-unificada_pt-BR.csv', input_path)

    uaa_df = pd.read_csv(uaa_path[0], delimiter=r',', na_filter=True) 
    ufa_df = pd.read_csv(ufa_path[0], delimiter=r',', na_filter=True)
    use_df = pd.read_csv(use_path[0], delimiter=r',', na_filter=True)
    uaa_df.set_index('participant_code')

    if 'origin' not in uaa_df.columns:
        uaa_df['origin'] = 'NINA'

    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    ###############################################################################
    # 101
    # nome: equInjTbpiAgeClasY_v1
    # descrição: idade na lesão de plexo braquial

    # convert date strings to datetime type
    uaa_df['acquisitiondate'] = pd.to_datetime(uaa_df['acquisitiondate'])
    uaa_df['datInjTbpi'] = pd.to_datetime(uaa_df['datInjTbpi'])
    ufa_df['acquisitiondate'] = pd.to_datetime(ufa_df['acquisitiondate'])
    ufa_df['datInjTbpi'] = pd.to_datetime(ufa_df['datInjTbpi'])
    uaa_df['datBirthdate'] = pd.to_datetime(uaa_df['datBirthdate'])

    # calculate and categorize the age (in years) of the patient when he/she acquired the lesion
    uaa_df['InjTbpiAge'] = uaa_df['datInjTbpi'] - uaa_df['datBirthdate']
    uaa_df['InjTbpiAge'] = uaa_df['InjTbpiAge'].dt.days.div(365.25) 

    uaa_df['equInjTbpiAgeClasY_v1'] = pd.cut(
        uaa_df['InjTbpiAge'],
        [-np.inf, 20, 30, 40, 50, 60, np.inf],
        labels=['<20', '20_29', '30_39','40_49','50_59','>=60']
    )

    # Faz equInjTbpiAgeClasY_v1 ser NINA quando datInjTbpi ou datBirthdate for vazio
    uaa_df['equInjTbpiAgeClasY_v1'] = np.where(np.isnan(uaa_df['InjTbpiAge'].values), 'NINA', uaa_df['equInjTbpiAgeClasY_v1'])

    uaa_df_columns = ['participant_code','equInjTbpiAgeClasY_v1']


    ###############################################################################
    # 102
    # nome: lstInjTpbiSide
    # descrição: lado da lesão de plexo braquial

    # just include the column; no need to make any transformation 
    uaa_df_columns.append('lstInjTpbiSide')

    ###############################################################################
    # 104
    # nome: mulPexDiagnosis_v1
    # descrição: diagnóstico da lesão de plexo

    def pexDiagnosis(response):
        if response['mulPexRDiagnosis[RComplete]'] == 'Y' or response['mulPexLDiagnosis[LComplete]'] == 'Y':
            return 'Complete'

        if response['mulPexRDiagnosis[RUpperTrunk]'] == 'Y' or response['mulPexLDiagnosis[LUpperTrunk]'] == 'Y':
            return 'Upper'

        if response['mulPexRDiagnosis[RExtendedUpperTrunk]'] == 'Y' or response['mulPexLDiagnosis[LExtendedUpperTrunk]'] == 'Y':
            return 'Extended'

        if response['mulPexRDiagnosis[RLowerTrunk]'] == 'Y' or response['mulPexLDiagnosis[LLowerTrunk]'] == 'Y':
            return 'Lower' 

        if response['mulPexRDiagnosis[NAIA]'] == 'Y' or response['mulPexLDiagnosis[NAIA]'] == 'Y':
            return 'NINA'   

        return 'Infra' 

    diagnosticColumns = ['mulPexRDiagnosis[RComplete]', 'mulPexLDiagnosis[LComplete]', 'mulPexRDiagnosis[RUpperTrunk]', 'mulPexLDiagnosis[LUpperTrunk]', 'mulPexRDiagnosis[RExtendedUpperTrunk]', 'mulPexLDiagnosis[LExtendedUpperTrunk]', 'mulPexRDiagnosis[RLowerTrunk]', 'mulPexLDiagnosis[LLowerTrunk]', 'mulPexRDiagnosis[NAIA]', 'mulPexLDiagnosis[NAIA]']

    uaa_df['mulPexDiagnosis_v1'] = uaa_df[diagnosticColumns].apply(lambda x: pexDiagnosis(x), axis=1)
    uaa_df_columns.append('mulPexDiagnosis_v1')


    ###############################################################################
    # 105
    # nome: mulInjTbpiCauseSide[Motorcycle]_v1
    # descrição: lesão por motocicleta

    def injTbpiCauseSideMotorcycle(response):
        if response['mulInjRTbpiCause[RMotorcycle]'] == 'Y' or response['mulInjLTbpiCause[LMotorcycle]'] == 'Y':
            return 'Y'

        if response['mulInjRTbpiCause[NINA]'] == 'Y' or response['mulInjLTbpiCause[NINA]'] == 'Y':
            return 'NINA'

        return 'N' 

    motorcycleColumns = ['mulInjRTbpiCause[RMotorcycle]', 'mulInjLTbpiCause[LMotorcycle]', 'mulInjRTbpiCause[NINA]', 'mulInjLTbpiCause[NINA]']

    uaa_df['mulInjTbpiCauseSide[Motorcycle]_v1'] = uaa_df[motorcycleColumns].apply(lambda x: injTbpiCauseSideMotorcycle(x), axis=1)
    uaa_df_columns.append('mulInjTbpiCauseSide[Motorcycle]_v1')

    ###############################################################################
    # 106
    # nome: lstInjSkuBra_v2
    # descrição: lesão cranioiencefálica acompanhando a LTPB

    def lstInjSkuBra(response):
        if 'Y' in list(response[['mulInjFractureSite[Skull]','mulInjOrtsurgSite[Skull]', 'yonInjTbi', 'yonInjUnconscio', 'yonInjBrainsurg']]):
            return 'Y'

        if 'NINA' in list(response[['yonInjFracture','yonInjOrtsurg','yonInjTbi', 'yonInjUnconscio', 'yonInjBrainsurg']]) or \
            'Y' in list(response[['mulInjFractureSite[NINA]', 'mulInjOrtsurgSite[NINA]']]):
            return 'NINA'

        return 'N' 

    injSkullColumns = ['mulInjFractureSite[Skull]','mulInjOrtsurgSite[Skull]', 'yonInjTbi', 
                    'yonInjUnconscio', 'yonInjBrainsurg', 'yonInjFracture', 'mulInjFractureSite[NINA]', 
                    'yonInjOrtsurg', 'mulInjOrtsurgSite[NINA]']

    uaa_df['lstInjSkuBra_v2'] = uaa_df[injSkullColumns].apply(lambda x: lstInjSkuBra(x), axis=1)
    uaa_df_columns.append('lstInjSkuBra_v2')


    ###############################################################################
    # 107
    # nome: lstInjUppLimScaGirSide_v2
    # descrição: lesão osteoarticular no membro superior e/ou cintura escapular do mesmo lado da LTPB 

    def lstInjUppLimScaGirSide(response):
        if response['lstInjTpbiSide'] == 'NINA':
            return 'NINA'

        resp = list(response[['lstInjFractureULimb','lstInjFractureScapul','lstInjFractureClavic','lstInjDislocSide',
                                'lstInjOrtsurgULimb','lstInjOrtsurgScapula','lstInjOrtsurgClavicu']])
        if response['lstInjTpbiSide'] == 'R' and ('R' in resp or 'RL' in resp):
                return 'Y'

        if response['lstInjTpbiSide'] == 'L' and ('L' in resp or 'RL' in resp):
                return 'Y'

        if response['lstInjTpbiSide'] == 'RL' and ('R' in resp or 'L' in resp or 'RL' in resp):
                return 'Y'
            
        resp1 = list(response[['yonInjFracture', 'lstInjFractureULimb','lstInjFractureScapul','lstInjFractureClavic',
                                'yonInjDisloc','lstInjDislocSide','yonInjOrtsurg', 'lstInjOrtsurgULimb',
                                'lstInjOrtsurgScapula','lstInjOrtsurgClavicu']])
        resp2 = list(response[['mulInjFractureSite[NINA]','mulInjOrtsurgSite[NINA]']])
        if 'NINA' in resp1 or 'Y' in resp2:
            return 'NINA'
        
        return 'N'
    

    injUppLimScaGirSideColumns = ['lstInjTpbiSide','lstInjFractureULimb','lstInjFractureScapul','lstInjFractureClavic',
                            'lstInjDislocSide','lstInjOrtsurgULimb','lstInjOrtsurgScapula','lstInjOrtsurgClavicu',
                            'yonInjFracture', 'mulInjFractureSite[NINA]', 'yonInjDisloc','yonInjOrtsurg', 'mulInjOrtsurgSite[NINA]']

    uaa_df['lstInjUppLimScaGirSide_v2'] = uaa_df[injUppLimScaGirSideColumns].apply(lambda x: lstInjUppLimScaGirSide(x), axis=1)
    uaa_df_columns.append('lstInjUppLimScaGirSide_v2')

    # Create the dataframe for the selected and processed data
    processed_df = uaa_df[uaa_df_columns].copy()
    processed_df.set_index('participant_code')

    # calculate the age of the lesion when the patient was evaluated 
    uaa_df['time_Acq_Tbpi'] = uaa_df['acquisitiondate'] - uaa_df['datInjTbpi']
    uaa_df['time_Acq_Tbpi'] = uaa_df['time_Acq_Tbpi'].dt.days.div(365.25)
    ufa_df['time_Acq_Tbpi'] = ufa_df['acquisitiondate'] - ufa_df['datInjTbpi']
    ufa_df['time_Acq_Tbpi'] = ufa_df['time_Acq_Tbpi'].dt.days.div(365.25)

    uaa_df['quest'] = 'UAA'
    ufa_df['quest'] = 'UFA'

    selected_columns = ['participant_code', 'time_Acq_Tbpi', 'lstPexMuscstrengthR[RShoulderAbduc]', 'lstPexMuscstrengthL[LShoulderAbduc]', 
    'lstPexMuscstrengthR[RShoulderExrot]', 'lstPexMuscstrengthL[LShoulderExrot]', 'lstPexMuscstrengthR[RElbowFlex]', 'lstPexMuscstrengthL[LElbowFlex]', 
    'yonPexPain', 'yonInjPhysio', 'yonInjOrthesis', 'yonInjMedicatio', 'txtInjMedicatioList[Opioids_Indication]', 
    'txtInjMedicatioList[Antidepressants_Indication]', 'txtInjMedicatioList[Anticonvulsants_Indication]', 'txtInjMedicatioList[Neuroleptics_Indication]',
    'txtInjMedicatioList[SupplementsVitamins_Indication]', 'txtInjMedicatioList[NaturalMedicinesTeas_Indication]',
    'txtInjMedicatioList[OtherClassMedicatio1_Indication]', 'txtInjMedicatioList[OtherClassMedicatio2_Indication]',
    'txtInjMedicatioList[OtherClassMedicatio3_Indication]', 'txtInjMedicatioList[OtherClassMedicatio4_Indication]',
        'txtInjMedicatioList[OtherClassMedicatio5_Indication]', 'lstInjOrthesisSide', 'quest']

    uaa_ufa = ufa_df[selected_columns]

    # add the column lstInjTpbiSide in the UFA responses
    uaa_ufa = uaa_ufa.merge(uaa_df[['participant_code', 'lstInjTpbiSide', 'origin']], how='inner', on='participant_code')

    # put together data from UFA and UAA
    uaa_ufa = pd.concat([uaa_ufa,uaa_df[selected_columns + ['lstInjTpbiSide', 'origin']]])


    # ###############################################################################
    # 209
    # nome:nome: yonPexPainTime1_v2
    # descrição: dor no período entre 0 e 6 meses (incluído) após a lesão de plexo
    # Questionários e preenchimentos considerados: AEU e ASU: todos os preenchimentos disponiveis dentro do periodo de tempo de interesse: 0 e 6 meses (incluído)
    #
    # Regra de desempate: se houver mais de um preenchimento para o periodo de interesse, considerar todos e usar as seguintes regras, na ordem apresentada, para gerar uma única resposta representativa do período em questão: 
    # 1ª) se em pelo menos um preenchimento, a resposta para yonPexPain for 'Y', considerar 'Y'
    # 2ª) se em pelo menos um preenchimento, a resposta para yonPexPain for 'NINA', considerar 'NINA'
    # 3ª) se em pelo menos um preenchimento, a resposta para yonPexPain for 'N', considerar 'N'
    # 4ª) se nenhuma das regras acima for preenchida, considerar 'NINA'  (inclusive para os pacientes que não tem respostas no período considerado)

    def yonPexPain(response):
        if 'Y' in response['yonPexPain']:
            return 'Y'

        if 'NINA' in response['yonPexPain']:
            return 'NINA'
        
        if 'N' in response['yonPexPain']:
            return 'N'

        return 'NINA'

    uaa_ufa_time1 = uaa_ufa[(uaa_ufa['time_Acq_Tbpi'] >= 0) & (uaa_ufa['time_Acq_Tbpi'] <= 0.5)].copy()

    if uaa_ufa_time1.shape[0] > 0:
        # Group the values in the yonPexPain column by patient
        uaa_ufa_yonPexPainTime1_v2 = uaa_ufa_time1.groupby(['participant_code'])['yonPexPain'].apply(list).reset_index(name='yonPexPain')
        uaa_ufa_yonPexPainTime1_v2['yonPexPainTime1_v2'] = uaa_ufa_yonPexPainTime1_v2.apply(lambda x: yonPexPain(x), axis=1) 

        processed_df = processed_df.merge(uaa_ufa_yonPexPainTime1_v2[['participant_code','yonPexPainTime1_v2']], how = 'left', on = 'participant_code')
        # Fill 'NINA' for the patients who has not responses in the time period of interest
        processed_df['yonPexPainTime1_v2'].fillna('NINA', inplace=True)
    else:
        # None of the patients has responses in the time period of interest, 
        # so all of them receive 'NINA' for 'yonPexPainTime1_v2'
        processed_df['yonPexPainTime1_v2'] = 'NINA'

    # ###############################################################################
    # 210
    # nome:nome: yonPexPainTime2_v2
    # descrição: dor no período entre 6 e 12 meses (incluído) após a lesão de plexo
    #
    # Regra de desempate: se houver mais de um preenchimento para o periodo de interesse, considerar todos e usar as seguintes regras, na ordem apresentada, para gerar uma única resposta representativa do período em questão: 
    # 1ª) se em pelo menos um preenchimento, a resposta para yonPexPain for 'Y', considerar 'Y'
    # 2ª) se em pelo menos um preenchimento, a resposta para yonPexPain for 'NINA', considerar 'NINA'
    # 3ª) se em pelo menos um preenchimento, a resposta para yonPexPain for 'N', considerar 'N'
    # 4ª) se nenhuma das regras acima for preenchida, considerar 'NINA'  (inclusive para os pacientes que não tem respostas no período considerado)

    uaa_ufa_time2 = uaa_ufa.loc[(uaa_ufa['time_Acq_Tbpi'] > 0.5) & (uaa_ufa['time_Acq_Tbpi'] <= 1)].copy()

    if uaa_ufa_time2.shape[0] > 0:
        # Group the values in the yonPexPain column by patient
        uaa_ufa_yonPexPainTime2_v2 = uaa_ufa_time2.groupby(['participant_code'])['yonPexPain'].apply(list).reset_index(name='yonPexPain')

        uaa_ufa_yonPexPainTime2_v2['yonPexPainTime2_v2'] = uaa_ufa_yonPexPainTime2_v2.apply(lambda x: yonPexPain(x), axis=1) 

        processed_df = processed_df.merge(uaa_ufa_yonPexPainTime2_v2[['participant_code','yonPexPainTime2_v2']], how = 'left', on = 'participant_code')

        # Fill 'NINA' for the patients who has not responses in the time period of interest
        processed_df['yonPexPainTime2_v2'].fillna('NINA', inplace=True)
    else:
        # None of the patients has responses in the time period of interest, 
        # so all of them receive 'NINA' for 'yonPexPainTime2_v2'
        processed_df['yonPexPainTime2_v2'] = 'NINA'

    # ###############################################################################
    # 301
    # nome: yonInjPhysio
    # descrição: tratamento com fisioterapia
    # Questionários e preenchimentos considerados: AEU e ASU: todos os preenchimentos feitos até 12 meses (incluido) após a lesão
    #
    # regras de desempate para pacientes com mais de um preenchimento:
    # 1ª) se em pelo menos um preenchimento, a resposta para yonInjPhysio for 'Y', considerar 'Y'
    # 2ª) se em pelo menos um preenchimento, a resposta para yonInjPhysio for 'NINA', considerar 'NINA'
    # 3ª) se em pelo menos um preenchimento, a resposta para yonInjPhysio for 'N', considerar 'N'
    # 4ª) se nenhuma das regras acima for preenchida, considerar 'NINA' (inclusive para pacientes sem preenchimentos no período de interesse)
    
    # ###############################################################################
    # 302
    # nome: yonInjOrthesisSide_v2
    # descrição: tratamento com órtese
    # Questionários e preenchimentos considerados: AEU e ASU: todos os preenchimentos feitos até 12 meses (incluido) após a lesão
    #
    # regras de desempate para pacientes com mais de um preenchimento:
    # 1ª) se em pelo menos um preenchimento, a resposta para yonInjOrthesisSide_v2 for 'Y', considerar 'Y'
    # 2ª) se em pelo menos um preenchimento, a resposta para yonInjOrthesisSide_v2 for 'NINA', considerar 'NINA'
    # 3ª) se em pelo menos um preenchimento, a resposta para yonInjOrthesisSide_v2 for 'N', considerar 'N'
    # 4ª) se nenhuma das regras acima for preenchida, considerar 'NINA' (inclusive quando o paciente não tem preenchimentos no período de interesse)


    def yonInjPhysio(response):
        if 'Y' in response['yonInjPhysio']:
            return 'Y'
            
        elif 'NINA' in response['yonInjPhysio']:
            return 'NINA'

        elif 'N' in response['yonInjPhysio']:
            return 'N'        

        return 'NINA'

    def yonInjOrthesisSide(response):
        if 'Y' in response['yonInjOrthesisSide_v2']:
            return 'Y'
            
        elif 'NINA' in response['yonInjOrthesisSide_v2']:
            return 'NINA'

        elif 'N' in response['yonInjOrthesisSide_v2']:
            return 'N'

        return 'NINA'

    uaa_ufa_time = uaa_ufa[(uaa_ufa['time_Acq_Tbpi'] >= 0) & (uaa_ufa['time_Acq_Tbpi'] <= 1)].copy()

    if uaa_ufa_time.shape[0] > 0:
        ### Process the yonInjPhysio attribute

        # Group the values in the yonInjPhysio column by patient
        uaa_ufa_yonInjPhysio = uaa_ufa_time.groupby(['participant_code'])['yonInjPhysio'].apply(list).reset_index(name='yonInjPhysio')
        uaa_ufa_yonInjPhysio['yonInjPhysio'] = uaa_ufa_yonInjPhysio.apply(lambda x: yonInjPhysio(x), axis=1)

        processed_df = processed_df.merge(uaa_ufa_yonInjPhysio[['participant_code','yonInjPhysio']], how = 'left', on = 'participant_code')

        ### Process the yonInjOrthesisSide_v2 attribute
        uaa_ufa_time['yonInjOrthesisSide_v2'] = 'N'
        uaa_ufa_time.loc[(uaa_ufa_time['yonInjOrthesis'] == 'NINA') | ( uaa_ufa_time['lstInjOrthesisSide'] == 'NINA') | ( uaa_ufa_time['lstInjTpbiSide'] == 'NINA') ,'yonInjOrthesisSide_v2'] = 'NINA'
        uaa_ufa_time.loc[(uaa_ufa_time['lstInjOrthesisSide'] == 'R') & uaa_ufa_time['lstInjTpbiSide'].isin(['R','RL']),'yonInjOrthesisSide_v2'] = 'Y'
        uaa_ufa_time.loc[(uaa_ufa_time['lstInjOrthesisSide'] == 'L') & uaa_ufa_time['lstInjTpbiSide'].isin(['L','RL']),'yonInjOrthesisSide_v2'] = 'Y'
        uaa_ufa_time.loc[(uaa_ufa_time['lstInjOrthesisSide'] == 'RL') & uaa_ufa_time['lstInjTpbiSide'].isin(['R','L','RL']),'yonInjOrthesisSide_v2'] = 'Y'

        # Group the values in the yonInjOrthesisSide_v2 column by patient
        uaa_ufa_yonInjOrthesis = uaa_ufa_time.groupby(['participant_code'])['yonInjOrthesisSide_v2'].apply(list).reset_index(name='yonInjOrthesisSide_v2')
        uaa_ufa_yonInjOrthesis['yonInjOrthesisSide_v2'] = uaa_ufa_yonInjOrthesis.apply(lambda x: yonInjOrthesisSide(x), axis=1)

        processed_df = processed_df.merge(uaa_ufa_yonInjOrthesis[['participant_code','yonInjOrthesisSide_v2']], how = 'left', on = 'participant_code')

        # Fill 'NINA' for the patients who has not responses in the time period of interest
        processed_df['yonInjPhysio'].fillna('NINA', inplace=True)
        processed_df['yonInjOrthesisSide_v2'].fillna('NINA', inplace=True)

    else:
        # None of the patients has responses in the time period of interest, 
        # so all of them receive 'NINA' for both attributes
        processed_df['yonInjPhysio'] = 'NINA'
        processed_df['yonInjOrthesisSide_v2'] = 'NINA'


    ###############################################################################
    # 401
    # nome: Bpsurgery_yon_type
    # descrição: cirurgia realizada
    # Questionários e preenchimentos considerados: AEU, ASU e ACU: todos os preenchimentos feitos até 12 meses (incluido) após a lesão


    def Bpsurgery(response):
        if response['mulBpsurgeryType[Transf]'] == 'Y' and 'Y' in response['mulBpsurgeryTypeList']:
            return 'Bpsurg_Y_Transf_Others'

        if response['mulBpsurgeryType[Transf]'] == 'Y' and response['mulBpsurgeryTypeList'] == ['N','N','N']:
            return 'Bpsurg_Y_Transf_Only'
        
        if response['mulBpsurgeryType[Transf]'] == 'N' and 'Y' in response['mulBpsurgeryTypeList']:
            return  'Bpsurg_Y_Others_Only'
        
        return 'Bpsurg_Y_Type_NINA'

    def BpsurgeryYonInjBpsurg(response):
        # Verifies the response for the field yonInjBpsurg of UAA or UFA
        if response == 'Y':
            return 'Bpsurg_Y_Type_NINA'

        if response == 'NINA':
            return 'Bpsurg_NINA'

        return 'Bpsurg_N'

    def BpsurgeryCombined(response):
        if 'Bpsurg_Y_Transf_Others' in response \
            or ('Bpsurg_Y_Transf_Only' in response \
                and 'Bpsurg_Y_Others_Only' in response):
            return 'Bpsurg_Y_Transf_Others'

        # Possible results ordered by priority
        ordered_results = ['Bpsurg_Y_Transf_Only', 'Bpsurg_Y_Others_Only', 'Bpsurg_Y_Type_NINA', 'Bpsurg_NINA', 'Bpsurg_N']
        for result  in ordered_results:
            if result in response: 
                return result
        
        return np.nan 


    # From now on, responses for the USE questionnaire will be used too
    use_df['datBpinjuryDate'] = pd.to_datetime(use_df['datBpinjuryDate'])
    use_df['datBpsurgDate'] = pd.to_datetime(use_df['datBpsurgDate'])
    use_df['time_Bpsurgery'] = use_df['datBpsurgDate'] - use_df['datBpinjuryDate']
    use_df['time_Bpsurgery'] = use_df['time_Bpsurgery'].dt.days.div(365.25)

    use_time = use_df[(use_df['time_Bpsurgery'] >= 0) & (use_df['time_Bpsurgery'] <= 1)].copy()
    uaa_time = uaa_df[(uaa_df['time_Acq_Tbpi'] >= 0) & (uaa_df['time_Acq_Tbpi'] <= 1)].copy()
    ufa_time = ufa_df[(ufa_df['time_Acq_Tbpi'] >= 0) & (ufa_df['time_Acq_Tbpi'] <= 1)].copy()

    # Join the values of the 'mulBpsurgeryType[XXX]' columns in a same new column called 'mulBpsurgeryTypeList'
    mulBpsurgeryType_columns = ['mulBpsurgeryType[Lysis]', 'mulBpsurgeryType[Graft]', 'mulBpsurgeryType[Neuromad]']
    use_time['mulBpsurgeryTypeList'] = use_time[mulBpsurgeryType_columns].apply(list, axis=1)

    # For each questionnarie response in the time interval of interest, get the value for the Bpsurgery_yon_type attribute
    use_time['Bpsurgery_yon_type'] = use_time.apply(lambda x: Bpsurgery(x), axis=1)
    uaa_time['Bpsurgery_yon_type'] = uaa_time['yonInjBpsurg'].apply(lambda x: BpsurgeryYonInjBpsurg(x))
    ufa_time['Bpsurgery_yon_type'] = ufa_time['yonInjBpsurg'].apply(lambda x: BpsurgeryYonInjBpsurg(x))

    # Group the values in the Bpsurgery_yon_type column by participant and combine them 
    use_Bpsurgery = use_time.groupby(['participant_code'])['Bpsurgery_yon_type'].apply(list).reset_index(name='Bpsurgery_yon_type')
    use_Bpsurgery['Bpsurgery_yon_type'] = use_Bpsurgery['Bpsurgery_yon_type'].apply(lambda x: BpsurgeryCombined(x))
    uaa_Bpsurgery = uaa_time.groupby(['participant_code'])['Bpsurgery_yon_type'].apply(list).reset_index(name='Bpsurgery_yon_type')
    uaa_Bpsurgery['Bpsurgery_yon_type'] = uaa_Bpsurgery['Bpsurgery_yon_type'].apply(lambda x: BpsurgeryCombined(x))
    ufa_Bpsurgery = ufa_time.groupby(['participant_code'])['Bpsurgery_yon_type'].apply(list).reset_index(name='Bpsurgery_yon_type')
    ufa_Bpsurgery['Bpsurgery_yon_type'] = ufa_Bpsurgery['Bpsurgery_yon_type'].apply(lambda x: BpsurgeryCombined(x))

    # Put the Bpsurgery data of UAA, UFA and USE in a same dataframe
    df_Bpsurgery = pd.concat([uaa_Bpsurgery[['participant_code','Bpsurgery_yon_type']],ufa_Bpsurgery[['participant_code','Bpsurgery_yon_type']], use_Bpsurgery[['participant_code','Bpsurgery_yon_type']]])

    # Group the values in the Bpsurgery_yon_type column by participant and combine them 
    df_Bpsurgery = df_Bpsurgery.groupby(['participant_code'])['Bpsurgery_yon_type'].apply(list).reset_index(name='Bpsurgery_yon_type')
    df_Bpsurgery['Bpsurgery_yon_type'] = df_Bpsurgery['Bpsurgery_yon_type'].apply(lambda x: BpsurgeryCombined(x))

    # Replace 'Bpsurg_NINA' by 'NINA'
    df_Bpsurgery['Bpsurgery_yon_type'].replace({'Bpsurg_NINA':'NINA'}, inplace=True)

    processed_df = processed_df.merge(df_Bpsurgery[['participant_code','Bpsurgery_yon_type']], how = 'left', on = 'participant_code')

    processed_df['Bpsurgery_yon_type'].fillna('NINA', inplace=True)


    ###############################################################################
    # 402
    # nome: Status_Explor_v1
    # descrição: status de exploração cirúrgica
      # Questionários e preenchimentos considerados: AEU, ASU e ACU: todos os preenchimentos feitos até 12 meses (incluido) após a lesão


    def StatusExplor(response):
        if response['yonExplor'] == 'Y':
            return 'Bpsurg_Y_Explor_Y'
        
        if response['yonExplor'] == 'NINA':
            return 'Bpsurg_Y_Explor_NINA'   
        
        return 'Bpsurg_Y_Explor_N'


    def StatusExplorYonInjBpsurg(response):
        if response == 'Y':
            return 'Bpsurg_Y_Explor_NINA'
        
        if response == 'NINA':
            return 'Bpsurg_NINA'   
        
        return 'Bpsurg_N'

    def StatusExplorCombined(response):

        ordered_results = ['Bpsurg_Y_Explor_Y', 'Bpsurg_Y_Explor_NINA', 'Bpsurg_Y_Explor_N', 'Bpsurg_NINA', 'Bpsurg_N'] 
        for result in ordered_results:
            if result in response:
                return result

        return np.nan


    def StatusExplorCombined2(response):
        ordered_results = ['Bpsurg_Y_Explor_Y', 'Bpsurg_Y_Explor_N', 'Bpsurg_Y_Explor_NINA', 'Bpsurg_NINA', 'Bpsurg_N'] 

        for result in ordered_results:
            if result in response:
                return result

        return np.nan


    use_time['Status_Explor_v1'] = use_time.apply(lambda x: StatusExplor(x), axis=1)
    uaa_time['Status_Explor_v1'] = uaa_time['yonInjBpsurg'].apply(lambda x: StatusExplorYonInjBpsurg(x))
    ufa_time['Status_Explor_v1'] = ufa_time['yonInjBpsurg'].apply(lambda x: StatusExplorYonInjBpsurg(x))

    # Group the values in the Bpsurgery column by participant
    use_Status_Explor = use_time.groupby(['participant_code'])['Status_Explor_v1'].apply(list).reset_index(name='Status_Explor_v1')
    use_Status_Explor['Status_Explor_v1'] = use_Status_Explor['Status_Explor_v1'].apply(lambda x: StatusExplorCombined(x))
    uaa_Status_Explor = uaa_time.groupby(['participant_code'])['Status_Explor_v1'].apply(list).reset_index(name='Status_Explor_v1')
    uaa_Status_Explor['Status_Explor_v1'] = uaa_Status_Explor['Status_Explor_v1'].apply(lambda x: StatusExplorCombined(x))
    ufa_Status_Explor = ufa_time.groupby(['participant_code'])['Status_Explor_v1'].apply(list).reset_index(name='Status_Explor_v1')
    ufa_Status_Explor['Status_Explor_v1'] = ufa_Status_Explor['Status_Explor_v1'].apply(lambda x: StatusExplorCombined(x))

    # Put the Bpsurgery data of UAA, UFA and USE in a same dataframe
    df_Status_Explor = pd.concat([uaa_Status_Explor[['participant_code','Status_Explor_v1']],ufa_Status_Explor[['participant_code','Status_Explor_v1']], use_Status_Explor[['participant_code','Status_Explor_v1']]])

    df_Status_Explor = df_Status_Explor.groupby(['participant_code'])['Status_Explor_v1'].apply(list).reset_index(name='Status_Explor_v1')
    df_Status_Explor['Status_Explor_v1'] = df_Status_Explor['Status_Explor_v1'].apply(lambda x: StatusExplorCombined2(x))

    # Replace 'Bpsurg_NINA' by 'NINA'
    df_Status_Explor['Status_Explor_v1'].replace({'Bpsurg_NINA':'NINA'}, inplace=True)
    
    processed_df = processed_df.merge(df_Status_Explor[['participant_code','Status_Explor_v1']], how = 'left', on = 'participant_code')
    processed_df['Status_Explor_v1'].fillna('NINA', inplace=True)


    ###############################################################################
    # 403
    # nome: Bpsurgery_time_v2
    # descrição: momento em que a primeira cirurgia foi realizada
 
    def Bpsurgery_time_combined(response):
        if '_6m' in response['Bpsurgery_time_v2']:  
            return '_6m'

        if '6_12m' in response['Bpsurgery_time_v2']: 
            return '6_12m'    

        if '12m_' in response['Bpsurgery_time_v2']:  
            return '12m_'

        return np.nan


    def Bpsurgery_time_combined2(response):
        if '_6m' in response['Bpsurgery_time_v2']:  
            return '_6m'

        if '6_12m' in response['Bpsurgery_time_v2']: 
            return '6_12m'    

        if '12m_' in response['Bpsurgery_time_v2']:  
            return '12m_'

        if 'Bpsurg_NINA' in response['Bpsurgery_time_v2']:  
            return 'Bpsurg_NINA'

        if 'Bpsurg_N' in response['Bpsurgery_time_v2']:  
            return 'Bpsurg_N'

        return np.nan


    # convert date strings to datetime type
    use_df['Bpsurgery_time_v2'] = use_df['time_Bpsurgery'].apply(lambda x: np.nan if x < 0 else ('_6m' if  x <= 0.5 else ('6_12m' if 0.5 < x <= 1 else '12m_')))
    uaa_df['Bpsurgery_time_v2'] = uaa_df['time_Acq_Tbpi'].apply(lambda x: np.nan if x < 0 else ('_6m' if  x <= 0.5 else ('6_12m' if 0.5 < x <= 1 else '12m_')))
    ufa_df['Bpsurgery_time_v2'] = ufa_df['time_Acq_Tbpi'].apply(lambda x: np.nan if x < 0 else ('_6m' if  x <= 0.5 else ('6_12m' if 0.5 < x <= 1 else '12m_')))
    uaa_df.loc[(uaa_df['yonInjBpsurg'] == 'NINA'),'Bpsurgery_time_v2'] = 'Bpsurg_NINA'
    uaa_df.loc[(uaa_df['yonInjBpsurg'] == 'N'),'Bpsurgery_time_v2'] = 'Bpsurg_N'
    uaa_df.loc[(uaa_df['yonInjBpsurg'].isna()),'Bpsurgery_time_v2'] = 'Bpsurg_N'
    ufa_df.loc[(ufa_df['yonInjBpsurg'] == 'NINA'),'Bpsurgery_time_v2'] = 'Bpsurg_NINA'
    ufa_df.loc[(ufa_df['yonInjBpsurg'] == 'N'),'Bpsurgery_time_v2'] = 'Bpsurg_N'
    ufa_df.loc[(ufa_df['yonInjBpsurg'].isna()),'Bpsurgery_time_v2'] = 'Bpsurg_N'
    
    # Group Bpsurgery_time_v2 by participant_code
    use_Bpsurgery_time = use_df.groupby(['participant_code'])['Bpsurgery_time_v2'].apply(list).reset_index(name='Bpsurgery_time_v2')
    use_Bpsurgery_time['Bpsurgery_time_v2'] = use_Bpsurgery_time.apply(lambda x: Bpsurgery_time_combined(x), axis=1)
    uaa_Bpsurgery_time = uaa_df.groupby(['participant_code'])['Bpsurgery_time_v2'].apply(list).reset_index(name='Bpsurgery_time_v2')
    uaa_Bpsurgery_time['Bpsurgery_time_v2'] = uaa_Bpsurgery_time.apply(lambda x: Bpsurgery_time_combined2(x), axis=1)
    ufa_Bpsurgery_time = ufa_df.groupby(['participant_code'])['Bpsurgery_time_v2'].apply(list).reset_index(name='Bpsurgery_time_v2')
    ufa_Bpsurgery_time['Bpsurgery_time_v2'] = ufa_Bpsurgery_time.apply(lambda x: Bpsurgery_time_combined2(x), axis=1)

   # Put the Bpsurgery_time_v2 data of UAA, UFA and USE in a same dataframe
    df_Bpsurgery_time_v2 = pd.concat([uaa_Bpsurgery_time[['participant_code','Bpsurgery_time_v2']],ufa_Bpsurgery_time[['participant_code','Bpsurgery_time_v2']], use_Bpsurgery_time[['participant_code','Bpsurgery_time_v2']]])

    df_Bpsurgery_time_v2 = df_Bpsurgery_time_v2.groupby(['participant_code'])['Bpsurgery_time_v2'].apply(list).reset_index(name='Bpsurgery_time_v2')
    df_Bpsurgery_time_v2['Bpsurgery_time_v2'] = df_Bpsurgery_time_v2.apply(lambda x: Bpsurgery_time_combined2(x), axis=1)

    # Replace 'Bpsurg_NINA' by 'NINA'
    df_Bpsurgery_time_v2['Bpsurgery_time_v2'].replace({'Bpsurg_NINA':'NINA'}, inplace=True)
    
    processed_df = processed_df.merge(df_Bpsurgery_time_v2[['participant_code','Bpsurgery_time_v2']], how = 'left', on = 'participant_code')
    
    processed_df['Bpsurgery_time_v2'].fillna('NINA', inplace=True)

    ###############################################################################
    # Demais atributos do tratamento cirurgico


    def mulTransfSiteGeneral(response):
        if response['mulTransfSite[NINA]'] == 'Y':
            return 'Transf_Y_Nv_NINA'

        if response['mulBpsurgeryType[Transf]'] == 'Y':  
            return 'Transf_Y_Nv_N'

        if response['mulBpsurgeryType[NINA]'] == 'Y':
            return 'Bpsurg_Y_Transf_NINA'

        return 'Bpsurg_Y_Transf_N'      


    def mulTransfSiteDonatorAC(response):
        if 'Y' in response['mulTransfSiteList']:  
            return 'Transf_Y_Nv_Y'      

        if 'acessÃ³rio' in response['txtTransfOther'] \
            or 'acessório' in response['txtTransfOther'] \
            or 'accessory' in response['txtTransfOther']: 
            return 'Transf_Y_Nv_Y'
    
        return mulTransfSiteGeneral(response) 


    def mulTransfSiteDonatorUl(response):
        if response['mulTransfSite[Oberlin]'] == 'Y' \
            or 'ulnar' in response['txtTransfOther'].lower():
            return 'Transf_Y_Nv_Y'      

        return mulTransfSiteGeneral(response) 


    def mulTransfSiteDonatorI(response):
        if response['mulTransfSite[IMc]'] == 'Y' \
            or response['mulTransfSite[IAx]'] == 'Y' \
            or 'intercostal' in response['txtTransfOther'].lower():
            return 'Transf_Y_Nv_Y'      

        return mulTransfSiteGeneral(response) 


    def mulTransfSiteDonatorPh(response):
        if response['mulTransfSite[PhMc]'] == 'Y' \
            or response['mulTransfSite[PhAc]'] == 'Y' \
            or 'frênico' in response['txtTransfOther'] \
            or 'frÃªnico' in response['txtTransfOther'] \
            or 'phrenic' in response['txtTransfOther']:
            return 'Transf_Y_Nv_Y'

        return mulTransfSiteGeneral(response)
        

    def mulTransfSiteReceptorSs(response):
        if response['mulTransfSite[AcSsAA]'] == 'Y' \
            or response['mulTransfSite[AcSsPA]'] == 'Y' \
            or 'supraescapular' in response['txtTransfOther'] \
            or 'suprascapular' in response['txtTransfOther'] :
            return 'Transf_Y_Nv_Y'    

        return mulTransfSiteGeneral(response) 


    def mulTransfSiteReceptorMc(response):
        if response['mulTransfSite[Oberlin]'] == 'Y' or response['mulTransfSite[IMc]'] == 'Y' \
            or response['mulTransfSite[IMc]'] == 'Y' or response['mulTransfSite[MPMc]'] == 'Y' \
            or response['mulTransfSite[AcMc]'] == 'Y' or response['mulTransfSite[MPMc]'] == 'Y' \
            or response['mulTransfSite[MPMc]'] == 'Y' or response['mulTransfSite[MPMc]'] == 'Y' \
            or response['mulTransfSite[PhMc]'] == 'Y' or response['mulTransfSite[MeMc]'] == 'Y' \
            or response['mulTransfSite[MeMc]'] == 'Y' or response['mulTransfSite[MPMc]'] == 'Y' \
            or response['mulTransfSite[MeBr]'] == 'Y' \
            or 'musculocutaneous' in response['txtTransfOther'] \
            or 'musculocutâneo' in response['txtTransfOther'] \
            or 'musculocutÃ¢neo' in response['txtTransfOther'] :

            return 'Transf_Y_Nv_Y'    

        return mulTransfSiteGeneral(response) 

    def mulTransfSiteReceptorAx(response):
        if response['mulTransfSite[MPAx]'] == 'Y' \
            or response['mulTransfSite[TrAx]'] == 'Y' \
            or response['mulTransfSite[IAx]'] == 'Y' \
            or 'axilar' in response['txtTransfOther'] \
            or 'axillary' in response['txtTransfOther'] :
            return 'Transf_Y_Nv_Y'    

        return mulTransfSiteGeneral(response) 


    def mulTransfSiteYonInjBpsurg(response):
        if response == 'Y':
            return 'Bpsurg_Y_Transf_NINA'

        if response == 'NINA':
            return 'Bpsurg_NINA'

        return 'Bpsurg_N'


    def mulTransfSiteCombined1(response):
        # possible results in order of priority
        results = ['Transf_Y_Nv_Y', 'Transf_Y_Nv_NINA', 'Transf_Y_Nv_N', 'Bpsurg_Y_Transf_NINA', 'Bpsurg_Y_Transf_N', 'Bpsurg_NINA', 'Bpsurg_N']
        for result in results:
            if result in response:
                return result

        return np.nan

    def mulTransfSiteCombined2(response):
        # possible results in order of priority
        results = ['Transf_Y_Nv_Y', 'Transf_Y_Nv_NINA', 'Transf_Y_Nv_N', 'Bpsurg_Y_Transf_N', 'Bpsurg_Y_Transf_NINA', 'Bpsurg_NINA', 'Bpsurg_N']
        for result in results:
            if result in response:
                return result

        return np.nan

    use_time['txtTransfOther'].fillna('', inplace=True)
    use_time['txtTransfOther'] = use_time['txtTransfOther'].str.lower()

    uaa_time['mulTransfSite'] = uaa_time['yonInjBpsurg'].apply(lambda x: mulTransfSiteYonInjBpsurg(x))
    ufa_time['mulTransfSite'] = ufa_time['yonInjBpsurg'].apply(lambda x: mulTransfSiteYonInjBpsurg(x))
    uaa_mulTransfSite = uaa_time.groupby(['participant_code'])['mulTransfSite'].apply(list).reset_index(name='mulTransfSite')
    uaa_mulTransfSite['mulTransfSite'] = uaa_mulTransfSite['mulTransfSite'].apply(lambda x: mulTransfSiteCombined1(x))
    ufa_mulTransfSite = ufa_time.groupby(['participant_code'])['mulTransfSite'].apply(list).reset_index(name='mulTransfSite')
    ufa_mulTransfSite['mulTransfSite'] = ufa_mulTransfSite['mulTransfSite'].apply(lambda x: mulTransfSiteCombined1(x))

    # Join the values of the 'mulTransfSite[XXX]' columns in a same new column called 'mulTransfSiteList'
    mulTransfSite_columns = ['mulTransfSite[AcSsAA]', 'mulTransfSite[AcSsPA]', 'mulTransfSite[AcMc]']
    use_time['mulTransfSiteList'] = use_time[mulTransfSite_columns].apply(list, axis=1)
    # Join the values of the 'mulBpsurgeryType[XXX]' columns in a same new column called 'mulBpsurgeryTypeList'
    mulBpsurgeryType_columns = ['mulBpsurgeryType[Lysis]', 'mulBpsurgeryType[Graft]', 'mulBpsurgeryType[Neuromad]']
    use_time['mulBpsurgeryTypeList'] = use_time[mulBpsurgeryType_columns].apply(list, axis=1)

    # Map of the mulTransfSite[XXX] attributes to be created and their correspondenting function 
    multTransfSiteAttributes = {
        "mulTransfSite[Donator_Ac]_v2" : mulTransfSiteDonatorAC,    
        "mulTransfSite[Donator_Ul]_v2" : mulTransfSiteDonatorUl,
        "mulTransfSite[Donator_I]_v2" : mulTransfSiteDonatorI,
        "mulTransfSite[Donator_Ph]_v2" : mulTransfSiteDonatorPh,
        "mulTransfSite[Receptor_Ss]_v2" : mulTransfSiteReceptorSs,
        "mulTransfSite[Receptor_Mc]_v2" : mulTransfSiteReceptorMc,
        "mulTransfSite[Receptor_Ax]_v2" : mulTransfSiteReceptorAx
    }

    for attribute, function in multTransfSiteAttributes.items():
        use_time[attribute] = use_time.apply(lambda x: function(x), axis=1)

        # Group values by participant 
        df_temp = use_time.groupby(['participant_code'])[attribute].apply(list).reset_index(name=attribute)
        df_temp[attribute] = df_temp[attribute].apply(lambda x: mulTransfSiteCombined1(x))

        # Put the mulTransfSite[XXX] colummn of UAA, UFA and USE in a same dataframe
        
        df_temp = pd.concat([uaa_mulTransfSite[['participant_code','mulTransfSite']].rename(columns={'mulTransfSite': attribute}),
                                ufa_mulTransfSite[['participant_code','mulTransfSite']].rename(columns={'mulTransfSite': attribute}), 
                                df_temp[['participant_code',attribute]]])

        df_temp = df_temp.groupby(['participant_code'])[attribute].apply(list).reset_index(name=attribute)
        df_temp[attribute] = df_temp[attribute].apply(lambda x: mulTransfSiteCombined2(x))

        df_temp[attribute].replace({'Bpsurg_NINA':'NINA'}, inplace=True)

        processed_df = processed_df.merge(df_temp[['participant_code',attribute]], how = 'left', on = 'participant_code')
    
        processed_df[attribute].fillna('Bpsurg_NINA', inplace=True)

    # TODO refactor the code to generate the attributes with their final categories (instead of changing the categories after the creation of the attributes)
    # Group categories in the surgical treatment data, to reduce to reduce data skew
    replacements = {"Bpsurg_Y_Transf_Others" : "Transf_Others",
                    "Bpsurg_Y_Transf_Only" : "Transf_Only",
                    "Bpsurg_Y_Others_Only" : "Others_Only",
                    "Bpsurg_Y_Type_NINA" : "NINA",
                    "Bpsurg_NINA" : "NINA",
                    "Bpsurg_N" : "N",
                    "Bpsurg_Y_Explor_Y" : "Y",
                    "Bpsurg_Y_Explor_N" : "N",
                    "Bpsurg_Y_Explor_NINA" : "NINA",
                    "Bpsurg_NINA" : "NINA",
                    "Bpsurg_N" : "N",
                    "Transf_Y_Nv_Y": "Y",
                    "Transf_Y_Nv_N": "N",
                    "Bpsurg_Y_Transf_N": "N",
                    "Bpsurg_Y_Transf_NINA": "NINA",
                    "Bpsurg_NINA": "NINA",
                    "Bpsurg_N": "N"}
    processed_df.replace(replacements, inplace = True)  



    if operation == OPERATION_TRAINING:
        processed_df.sort_values(by='participant_code').to_csv(os.path.join(output_path,"all_instances.csv"), index = False, quoting = csv.QUOTE_NONNUMERIC)
             
        processed_df = processed_df.merge(uaa_df[['participant_code','origin']], how = 'left', on = 'participant_code')
        select_training_instances(output_path, processed_df, uaa_ufa, 1.5, min, 'group1', False, data_reports)   # grupo 1: atual com desfecho após 1,5 anos (desempate pelo mais próximo de 1,5 anos)
        select_training_instances(output_path, processed_df, uaa_ufa, 1.5, max, 'group2', False, data_reports)   # grupo 2: com desfecho após 1,5 anos (desempate pelo mais distante de 1,5 anos)
        select_training_instances(output_path, processed_df, uaa_ufa, 3, min, 'group3', False, data_reports)   # grupo 3: atual com desfecho após 3 anos (desempate pelo mais próximo de 3 anos)
        select_training_instances(output_path, processed_df, uaa_ufa, 3, max, 'group4', False, data_reports)   # grupo 4: atual com desfecho após 3 anos (desempate pelo mais distante de 3 anos)
        select_training_instances(output_path, processed_df, uaa_ufa, 2, min, 'group5', False, data_reports)   # grupo 5: atual com desfecho após 2 anos (desempate pelo mais próximo de 2 anos)
        select_training_instances(output_path, processed_df, uaa_ufa, 2, max, 'group6', False, data_reports)   # grupo 6: atual com desfecho após 2 anos (desempate pelo mais distante de 2 anos)

    else: # data for classification
        if operation == OPERATION_BULK_CLASSIFICATION:
            file_name = 'bulk_classification_data.csv' 
        else:            
            file_name = participant_code + '_classification_data.csv'

        processed_df.to_csv(os.path.join(output_path,file_name), index = False, quoting = csv.QUOTE_NONNUMERIC)




"""
Select the instances of the training set for each one of the four prognostic models
"""
def select_training_instances(output_path, processed_df, uaa_ufa, time_Acq_Tbpi_threshold, 
                        proximity_function, group_name, only_ufa = False, data_reports = True):

    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    uaa_ufa = uaa_ufa[['participant_code', 'origin', 'lstInjTpbiSide', 'time_Acq_Tbpi', 
        'yonPexPain','lstPexMuscstrengthR[RShoulderAbduc]', 'lstPexMuscstrengthL[LShoulderAbduc]',
        'lstPexMuscstrengthR[RShoulderExrot]', 'lstPexMuscstrengthL[LShoulderExrot]', 
        'lstPexMuscstrengthR[RElbowFlex]', 'lstPexMuscstrengthL[LElbowFlex]','quest']].copy()

    def lstPexMuscstrength_Outcome(resp, rfield, lfield, time_Acq_Tbpi_threshold):
        rValue = str(resp[rfield])
        lValue = str(resp[lfield])
        values = ['0','1','2','3','4','5']

        if resp['time_Acq_Tbpi'] < time_Acq_Tbpi_threshold: 
            return 'N'
        if resp['lstInjTpbiSide'] == 'R' and rValue in values:
            return 'Y'
        if resp['lstInjTpbiSide'] == 'L' and lValue in values:
            return 'Y'
        if resp['lstInjTpbiSide'] == 'RL' and (rValue in values or lValue in values):
            return 'Y'

        return 'N'

    def yonPexPain_outcome(resp, time_Acq_Tbpi_threshold): 
        if resp['time_Acq_Tbpi'] < time_Acq_Tbpi_threshold:
            return 'N'

        if resp['yonPexPain'] in ['Y','N']:
            return 'Y'
        
        return 'N'


    def lstPexMuscstrengthBinaryOutcome(response, rfield, lfield):
        # Returns 1 if good outcome and 0 in the contrary case

        # categorize the shoulder abduction strength in the side of the lesion
        if response['lstInjTpbiSide'] == 'R':
            value = response[rfield]
        elif response['lstInjTpbiSide'] == 'L':
            value = response[lfield]
        elif response['lstInjTpbiSide'] == 'RL':
            response['lstInjTpbiSide'] = 'R'
            v1 = lstPexMuscstrengthBinaryOutcome(response, rfield, lfield)
            response['lstInjTpbiSide'] = 'L'
            v2 = lstPexMuscstrengthBinaryOutcome(response, rfield, lfield)
            return max(v1,v2)
        else:
            return np.nan

        if value in ['0','1','2']:
            return 0
            
        if value in ['3','4','5']:
            return 1 
        
        return np.nan


    def yonPexPainBinaryOutcome(response):
        # Returns 1 if good outcome and 0 in the contrary case

        if response['yonPexPain'] == 'Y':
            return 0
            
        if response['yonPexPain'] == 'N':
            return 1
        
        return np.nan

    # Identificar os pacientes que preenchem cada um dos critérios do desfecho, usando filtro		
    uaa_ufa['lstPexMuscstrength[ShoulderAbduc]Side_outcome'] = uaa_ufa.apply(lambda x: lstPexMuscstrength_Outcome(x, 'lstPexMuscstrengthR[RShoulderAbduc]', 'lstPexMuscstrengthL[LShoulderAbduc]', time_Acq_Tbpi_threshold), axis = 1)
    uaa_ufa['lstPexMuscstrength[ShoulderExrot]Side_outcome'] = uaa_ufa.apply(lambda x: lstPexMuscstrength_Outcome(x, 'lstPexMuscstrengthR[RShoulderExrot]', 'lstPexMuscstrengthL[LShoulderExrot]', time_Acq_Tbpi_threshold), axis = 1)
    uaa_ufa['lstPexMuscstrength[ElbowFlex]Side_outcome'] = uaa_ufa.apply(lambda x: lstPexMuscstrength_Outcome(x, 'lstPexMuscstrengthR[RElbowFlex]', 'lstPexMuscstrengthL[LElbowFlex]', time_Acq_Tbpi_threshold), axis = 1)
    uaa_ufa['yonPexPain_outcome'] = uaa_ufa.apply(lambda x: yonPexPain_outcome(x, time_Acq_Tbpi_threshold), axis = 1)

    uaa_ufa['lstPexMuscstrength_ShoulderAbduc'] = uaa_ufa.apply(lambda x: lstPexMuscstrengthBinaryOutcome(x, 'lstPexMuscstrengthR[RShoulderAbduc]', 'lstPexMuscstrengthL[LShoulderAbduc]'), axis=1)
    uaa_ufa['lstPexMuscstrength_ShoulderExrot'] = uaa_ufa.apply(lambda x: lstPexMuscstrengthBinaryOutcome(x, 'lstPexMuscstrengthR[RShoulderExrot]', 'lstPexMuscstrengthL[LShoulderExrot]'), axis=1)
    uaa_ufa['lstPexMuscstrength_ElbowFlex'] = uaa_ufa.apply(lambda x: lstPexMuscstrengthBinaryOutcome(x, 'lstPexMuscstrengthR[RElbowFlex]', 'lstPexMuscstrengthL[LElbowFlex]'), axis=1)
    uaa_ufa['yonPexPain'] = uaa_ufa.apply(lambda x: yonPexPainBinaryOutcome(x), axis=1)

    processed_df_with_outcomes = processed_df.copy(deep=True)

    outcome_tables = []
    outcomes = {'lstPexMuscstrength_ShoulderAbduc':'lstPexMuscstrength[ShoulderAbduc]Side_outcome',
                'lstPexMuscstrength_ShoulderExrot':'lstPexMuscstrength[ShoulderExrot]Side_outcome',
                'lstPexMuscstrength_ElbowFlex':'lstPexMuscstrength[ElbowFlex]Side_outcome',
                'yonPexPain':'yonPexPain_outcome'}

    
    if data_reports:    
        reports_path = os.path.join(output_path,'reports')
        Path(reports_path).mkdir(parents=True, exist_ok=True)
        df = processed_df.copy(deep=True)
        del(df['participant_code'])
        profile = ProfileReport(df, title="Data Profiling Report - All Instances with Outcomes", explorative=True,  minimal=False)
        profile.to_file(os.path.join(reports_path,"all_instances.html"))
        data = []
        labels = []

    # For each outcome
    for outcome, outcome_attr in outcomes.items():
        # Select the patients who satisfy the selection criteria
        if only_ufa:
            df_outcome = uaa_ufa[(uaa_ufa[outcome_attr] == 'Y') & (uaa_ufa['quest'] == 'UFA')]
        else:
            df_outcome = uaa_ufa[uaa_ufa[outcome_attr] == 'Y']
        
        # aplica o critério de desempate (para o caso do paciente ter mais de uma resposta)
        idx = df_outcome.groupby(['participant_code'])['time_Acq_Tbpi'].transform(proximity_function) == df_outcome['time_Acq_Tbpi']
        df_outcome = df_outcome[idx]
 
        if data_reports:
            data.append(df_outcome['time_Acq_Tbpi'])
            labels.append(outcome)

        # junta a coluna de outcome aos dados preprocessados do paciente
        processed_df_with_outcomes = processed_df_with_outcomes.merge(df_outcome[['participant_code', outcome]], how = 'left', on = 'participant_code')
        df_outcome = processed_df.merge(df_outcome[['participant_code',outcome]], how = 'right', on = 'participant_code')
        del[df_outcome['origin']]
        outcome_tables.append(df_outcome)

    if data_reports:
        fig, ax = plt.subplots()
        ax.set_title('Outcome time (in years)')
        ax.boxplot(data, showmeans = True, labels = labels)
        plt.savefig(os.path.join(reports_path,group_name+'_times.png'))
    

    # Save a file for each type of outcome 
    # Also, separate the data according to its origin
    origins = list(processed_df_with_outcomes['origin'].str.upper().unique())
    if len(origins) > 1:
        origins.append('')   # use this to create files containning data from all origins together

    outcome_tables_names = list(outcomes.keys())
    for i in range(0,len(outcome_tables)):
        df_outcome_table = outcome_tables[i]
        for origin in origins:
            codes = list(processed_df_with_outcomes[processed_df_with_outcomes['origin'].str.contains(origin)]['participant_code'])
            df = df_outcome_table[df_outcome_table['participant_code'].isin(codes)]
            if df.shape[0] > 0:
                if origin == '':
                    origin = 'all'
                else:
                    origin = origin.strip().replace(' ','_')

                dir_path = os.path.join(output_path,group_name,origin.lower())
                Path(dir_path).mkdir(parents=True, exist_ok=True)

                df.to_csv(os.path.join(dir_path, outcome_tables_names[i] + '_data.csv'), index = False, quoting = csv.QUOTE_NONNUMERIC)

                if data_reports:
                    dir_path = os.path.join(reports_path,group_name,origin.lower())
                    Path(dir_path).mkdir(parents=True, exist_ok=True)
                    profile = ProfileReport(df, title="Data Profiling Report - " + outcome_tables_names[i], explorative=True,  minimal=False)
                    profile.to_file(os.path.join(dir_path, outcome_tables_names[i] + "_data.html"))


    processed_df_with_outcomes.to_csv(os.path.join(output_path,group_name,'all_instances_with_outcomes.csv'), index = False, quoting = csv.QUOTE_NONNUMERIC)
    if data_reports:
        profile = ProfileReport(processed_df_with_outcomes, title="Data Profiling Report - " + outcome_tables_names[i], explorative=True,  minimal=False)
        profile.to_file(os.path.join(reports_path,group_name,outcome_tables_names[i] + "all_instances_with_outcomes.html"))


if __name__ == '__main__':
    preprocess('./questionnaire_data', './training_data2', operation=OPERATION_TRAINING, participant_code=None, data_reports=False)


