import pandas as pd

'''
Script para agregar as seguintes bases:
- Estabelecimentos
- Recursos Físicos
- Recursos Humanos
- Equipes
'''


def split_cod_name(df):
    df['cod_mun'] = df['Município'].apply(lambda x: x.split(' ')[0])
    df['nome_mun'] = df['Município'].apply(lambda x: ', '.join(x.split(' ')[1:]).replace(',', ''))

    return df


def split_cod_name_rf(df):
    df['cod_mun'] = df.iloc[:,0].apply(lambda x: x.split(' ')[0])
    df['nome_mun'] = df.iloc[:,0].apply(lambda x: ', '.join(x.split(' ')[1:]).replace(',', ''))

    return df


def group_by_year(df, db_name):
    df = df.replace('-', 0)
    df_float = df.filter(regex='2014|2015|2016|2017|2018').astype(float, inplace=True)
    df[db_name + '2014'] = df_float.filter(regex='2014').max(axis=1)
    df[db_name + '2015'] = df_float.filter(regex='2015').max(axis=1)
    df[db_name + '2016'] = df_float.filter(regex='2016').max(axis=1)
    df[db_name + '2017'] = df_float.filter(regex='2017').max(axis=1)
    df[db_name + '2018'] = df_float.filter(regex='2018').max(axis=1)
    del df_float

    return df


def processar_estabelecimentos(nrows, tipo_estabelecimento, prefixo_coluna):
    df_est = pd.read_csv(
        f'data/Banco_Datathon/Banco_Datathon/estabelecimentos/{tipo_estabelecimento}',
        encoding='ISO-8859-1',
        error_bad_lines=False,
        index_col=0,
        sep=';',
        skiprows=4,
        nrows=nrows
    )

    df_est = df_est.head(len(df_est) - 7) # Remove as última 8 colunas
    df_est.reset_index(inplace=True)
    df_est = split_cod_name(df_est) # Separa código do nome do município
    df_est = group_by_year(df_est, prefixo_coluna) # Agrupamento por ano
    df_est = df_est.filter(regex=f'^{prefixo_coluna}|cod_mun|nome_mun') # Filtra colunas que serão utilizadas
    print(df_est.shape)

    return df_est


def processar_recursos_fisicos(nrows, tipo_rf, prefixo_coluna):
    df_rf = pd.read_csv(
        f'data/Banco_Datathon/Banco_Datathon/recursos_fisicos/{tipo_rf}',
        encoding='ISO-8859-1',
        index_col=0,
        nrows=nrows,
        skiprows=3,
        sep=',',
        low_memory=True
    )
    df_rf.reset_index(inplace=True)
    df_rf = split_cod_name_rf(df_rf)  # Separa código do nome do município
    df_rf = group_by_year(df_rf, prefixo_coluna)  # Agrupamento por ano
    df_rf = df_rf.filter(regex=f'^{prefixo_coluna}|cod_mun|nome_mun')  # Filtra colunas que serão utilizadas
    print(df_rf.shape)

    return df_rf


def processar_recursos_humanos(nrows, tipo_rh, prefixo_coluna):
    df_rh = pd.read_csv(
        f'data/Banco_Datathon/Banco_Datathon/recursos_humanos/{tipo_rh}',
        encoding='ISO-8859-1',
        index_col=0,
        nrows=nrows,
        skiprows=4,
        sep=';',
        low_memory=True
    )
    df_rh = df_rh.head(len(df_rh) - 7)  # Remove as última 8 colunas
    df_rh.reset_index(inplace=True)
    df_rh = split_cod_name(df_rh)  # Separa código do nome do município
    df_rh = group_by_year(df_rh, prefixo_coluna)  # Agrupamento por ano
    df_rh = df_rh.filter(regex=f'^{prefixo_coluna}|cod_mun|nome_mun')  # Filtra colunas que serão utilizadas
    print(df_rh.shape)

    return df_rh


def processar_equipes(nrows, tipo_equipe, prefixo_coluna):
    df_equipes = pd.read_csv(
        f'data/Banco_Datathon/Banco_Datathon/equipes/{tipo_equipe}',
        encoding='ISO-8859-1',
        index_col=0,
        nrows=nrows,
        skiprows=4,
        sep=';',
        low_memory=True
    )
    df_equipes = df_equipes.head(len(df_equipes) - 7)  # Remove as última 8 colunas
    df_equipes.reset_index(inplace=True)
    df_equipes = split_cod_name(df_equipes)  # Separa código do nome do município
    df_equipes = group_by_year(df_equipes, prefixo_coluna)  # Agrupamento por ano
    df_equipes = df_equipes.filter(regex=f'^{prefixo_coluna}|cod_mun|nome_mun')  # Filtra colunas que serão utilizadas
    print(df_equipes.shape)

    return df_equipes


def main():
    nrows=100000

    '''
    ------------------------------------------------------------------------------------------------
    PROCESSA BASES DE ESTABELECIMENTOS
    ------------------------------------------------------------------------------------------------
    '''
    print(f'Processando base de estabelecimentos: ')

    arquivos_est = [
        'Estabelecimentos_Clínicas_Ambulatórios_Especializados.csv',
        'Estabelecimentos- Hospital Especializado.csv',
        'Estabelecimentos- Hospital Geral.csv',
        'Estabelecimentos- Unidade Básica de Saúde.csv',
        'Estabelecimentos- Unidade de Serviço de Apoio ao Diagnose e Terapia.csv'
    ]

    df_est_cli_amb_esp = processar_estabelecimentos(nrows, arquivos_est[0], 'est_cli_amb_esp_max_')
    df_hosp_esp = processar_estabelecimentos(nrows, arquivos_est[1], 'hosp_esp_max_')
    df_hop_geral = processar_estabelecimentos(nrows, arquivos_est[2], 'hop_geral_max_')
    df_ubs = processar_estabelecimentos(nrows, arquivos_est[3], 'ubs_max_')
    df_diagnose_e_terapia = processar_estabelecimentos(nrows, arquivos_est[4], 'diagnose_e_terapia_max_')

    df_est = pd.merge(df_est_cli_amb_esp, df_hosp_esp, how='inner', on='cod_mun')
    print(df_est.shape)
    df_est = pd.merge(df_est, df_hop_geral, how='inner', on='cod_mun')
    print(df_est.shape)
    df_est = pd.merge(df_est, df_ubs, how='inner', on='cod_mun')
    print(df_est.shape)
    df_est = pd.merge(df_est, df_diagnose_e_terapia, how='inner', on='cod_mun')
    print(df_est.shape)
    df_est.drop(columns=df_est.filter(regex='nome_mun_').columns.tolist(), inplace=True)
    print(df_est.shape)

    df_est.to_csv('data/Banco_Datathon/Banco_Datathon/processed/estabelecimentos.csv')
    del df_est

    '''
    ------------------------------------------------------------------------------------------------
    PROCESSA BASES DE RECURSOS FÍSICOS
    ------------------------------------------------------------------------------------------------
    '''
    print(f'Processando base de recursos físicos: ')

    arquivos_rf = [
        'RF- Leitos de Internação.csv',
        'RF- Mamógrafos.csv',
        'RF- Raios X.csv',
        'RF-Ressonância Magnética.csv',
        'RF- Tomógrafos Computadorizados.csv'
    ]

    df_rf_leitos = processar_recursos_fisicos(nrows, arquivos_rf[0], 'rf_leitos_')
    df_rf_mamografos = processar_recursos_fisicos(nrows, arquivos_rf[1], 'rf_mamografos_')
    df_rf_raios_x = processar_recursos_fisicos(nrows, arquivos_rf[2], 'rf_raios_x_')
    df_rf_ressonancia_mag = processar_recursos_fisicos(nrows, arquivos_rf[3], 'rf_ressonancia_mag_')
    df_rf_tomografos_comp = processar_recursos_fisicos(nrows, arquivos_rf[4], 'rf_tomografos_comp_')

    df_rf = pd.merge(df_rf_leitos, df_rf_mamografos, how='inner', on='cod_mun')
    print(df_rf.shape)
    df_rf = pd.merge(df_rf, df_rf_raios_x, how='inner', on='cod_mun')
    print(df_rf.shape)
    df_rf = pd.merge(df_rf, df_rf_ressonancia_mag, how='inner', on='cod_mun')
    print(df_rf.shape)
    df_rf = pd.merge(df_rf, df_rf_tomografos_comp, how='inner', on='cod_mun')
    print(df_rf.shape)
    df_rf.drop(columns=df_rf.filter(regex='nome_mun_').columns.tolist(), inplace=True)
    print(df_rf.shape)

    df_rf.to_csv('data/Banco_Datathon/Banco_Datathon/processed/recursos_fisicos.csv')
    del df_rf

    '''
    ------------------------------------------------------------------------------------------------
    PROCESSA BASES DE RECURSOS HUMANOS
    ------------------------------------------------------------------------------------------------
    '''
    print(f'Processando base de recursos humanos: ')

    arquivos_rh = ['RH- Enfermeiros.csv', 'RH- Médicos.csv']

    df_rh_enfermeiros = processar_recursos_humanos(nrows, arquivos_rh[0], 'rh_enfermeiros_')
    df_rh_medicos = processar_recursos_humanos(nrows, arquivos_rh[1], 'rh_medicos_')
    df_rh = pd.merge(df_rh_enfermeiros, df_rh_medicos, how='inner', on='cod_mun', suffixes=('', '_medicos'))
    print(df_rh.shape)
    df_rh.drop(columns=df_rh.filter(regex='nome_mun_medicos').columns.tolist(), inplace=True)
    print(df_rh.shape)

    df_rh.to_csv('data/Banco_Datathon/Banco_Datathon/processed/recursos_humanos.csv')
    del df_rh

    '''
    ------------------------------------------------------------------------------------------------
    PROCESSA BASES EQUIPES
    ------------------------------------------------------------------------------------------------
    '''
    print(f'Processando base de equipes: ')

    arquivos_equipes = ['Equipes de Saúde- Equipes Saúde da Familia.csv',
                        'Equipes de Saúde- Núcleos de Apoio à Saúde da Familia- NASF.csv']

    df_saude_da_familia = processar_equipes(nrows, arquivos_equipes[0], 'equipes_saude_da_familia_')
    df_nucleos_de_apoio_saude_da_familia = processar_equipes(nrows, arquivos_equipes[1], 'equipes_nucleos_de_apoio_saude_da_familia_')
    df_equipes = pd.merge(
        df_saude_da_familia, df_nucleos_de_apoio_saude_da_familia, how='inner', on='cod_mun', suffixes=('', '_nucleos')
    )
    print(df_equipes.shape)
    df_equipes.drop(columns=['nome_mun_nucleos'], inplace=True)
    print(df_equipes.shape)

    df_equipes.to_csv('data/Banco_Datathon/Banco_Datathon/processed/equipes.csv')
    del df_equipes


if __name__ == '__main__':
    main()
